import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

# ======================
# CONFIG
# ======================
DATASET_PATH = "dataset"
IMG_SIZE = 224            # must match IMG_SIZE in drproject/settings.py
BATCH_SIZE = 32
HEAD_EPOCHS = 8           # phase 1: train only the new classifier head
FINE_TUNE_EPOCHS = 40     # phase 2: fine-tune the top of the backbone
FINE_TUNE_LAYERS = 110
SEED = 42
MODEL_OUT = "dr_eye_disease_model.h5"

tf.keras.utils.set_random_seed(SEED)

# Alphabetical order == CLASS_NAMES in detector/ml_model.py
CLASS_NAMES = sorted(
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
)
print("Classes:", CLASS_NAMES)

# ======================
# LOAD IMAGES (originals only; augmentation happens on the fly, after the split)
# ======================
images, labels = [], []
for idx, category in enumerate(CLASS_NAMES):
    folder = os.path.join(DATASET_PATH, category)
    for name in tqdm(os.listdir(folder), desc=category):
        img = cv2.imread(os.path.join(folder, name))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)          # uint8, 0-255
        labels.append(idx)

images = np.array(images, dtype=np.uint8)
labels = np.array(labels)

# ======================
# SPLIT BEFORE AUGMENTING (no leakage): 70 / 15 / 15, stratified
# ======================
X_train, X_tmp, y_train, y_tmp = train_test_split(
    images, labels, test_size=0.30, random_state=SEED, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp)
print("train/val/test:", len(X_train), len(X_val), len(X_test))

# Balanced class weights so rare classes (Mild/Severe/Proliferate) count as much as No_DR
weights = compute_class_weight("balanced", classes=np.arange(len(CLASS_NAMES)), y=y_train)
class_weight = {i: float(w) for i, w in enumerate(weights)}
print("Class weights:", class_weight)

# ======================
# DATA PIPELINE
# ======================
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.15),
    layers.RandomBrightness(0.1, value_range=(0.0, 1.0)),
])


def make_ds(X, y, training):
    ds = tf.data.Dataset.from_tensor_slices((X, tf.one_hot(y, len(CLASS_NAMES))))
    if training:
        ds = ds.shuffle(len(X), seed=SEED)
    # model input convention used by the Django app: float32 in [0, 1]
    ds = ds.map(lambda a, b: (tf.cast(a, tf.float32) / 255.0, b),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    if training:
        ds = ds.map(lambda a, b: (tf.clip_by_value(augment(a, training=True), 0.0, 1.0), b),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


train_ds = make_ds(X_train, y_train, True)
val_ds = make_ds(X_val, y_val, False)
test_ds = make_ds(X_test, y_test, False)

# ======================
# MODEL: ImageNet MobileNetV2 backbone + new classifier head
# ======================
base = MobileNetV2(weights="imagenet", include_top=False,
                   input_shape=(IMG_SIZE, IMG_SIZE, 3), name="backbone")
base.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = layers.Rescaling(2.0, offset=-1.0, name="to_minus1_1")(inputs)  # [0,1] -> [-1,1]
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
model = models.Model(inputs, outputs)

class MacroF1Checkpoint(tf.keras.callbacks.Callback):
    """Keeps the weights with the best validation macro-F1, so the saved model
    is the one that is good on ALL classes, not just the frequent ones."""

    def __init__(self, ds, y_true, patience):
        super().__init__()
        self.ds, self.y_true, self.patience = ds, y_true, patience
        self.best, self.best_weights, self.wait = -1.0, None, 0

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(self.ds, verbose=0).argmax(1)
        f1 = f1_score(self.y_true, pred, average="macro")
        print(f" - val_macro_f1: {f1:.4f}")
        if f1 > self.best:
            self.best, self.best_weights, self.wait = f1, self.model.get_weights(), 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

    def restore(self):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            print(f"Restored best weights (val macro-F1 {self.best:.4f})")


def run_phase(lr, epochs, patience):
    model.compile(optimizer=Adam(lr), loss="categorical_crossentropy", metrics=["accuracy"])
    ckpt = MacroF1Checkpoint(val_ds, y_val, patience)
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weight,
              callbacks=[ckpt, tf.keras.callbacks.ReduceLROnPlateau(
                  monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)])
    ckpt.restore()
    return ckpt.best


print("Phase 1: training head")
run_phase(1e-3, HEAD_EPOCHS, patience=8)

print("Phase 2: fine-tuning")
base.trainable = True
for layer in base.layers[:-FINE_TUNE_LAYERS]:
    layer.trainable = False
for layer in base.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False
run_phase(1e-4, FINE_TUNE_EPOCHS, patience=10)

# ======================
# EVALUATE (held-out test split, per class)
# ======================
model.save(MODEL_OUT)
probs = model.predict(test_ds, verbose=0)
pred = probs.argmax(1)
report = classification_report(y_test, pred, target_names=CLASS_NAMES, digits=3)
cm = confusion_matrix(y_test, pred)
print(report)
print(cm)
with open("training_report.txt", "w") as f:
    f.write(f"Test accuracy: {(pred == y_test).mean() * 100:.2f}%\n\n{report}\n{cm}\n")
print("Model saved to", MODEL_OUT)
