import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import albumentations as A

# ======================
# CONFIG
# ======================
DATASET_PATH = "dataset"
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.0001
AUGMENT_TIMES = 1   # how many augmented copies per image

# ======================
# DATA AUGMENTATION PIPELINE
# ======================
augmentor = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
])

# ======================
# LOAD + PREPROCESS
# ======================
images = []
labels = []

print("📂 Loading & preprocessing images...")

for category in os.listdir(DATASET_PATH):
    category_path = os.path.join(DATASET_PATH, category)

    if not os.path.isdir(category_path):
        continue

    for img_name in tqdm(os.listdir(category_path), desc=f"Processing {category}"):
        img_path = os.path.join(category_path, img_name)

        try:
            # READ IMAGE
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # RESIZE
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # NOISE FILTERING
            img = cv2.GaussianBlur(img, (5, 5), 0)

            # NORMALIZATION
            img = img / 255.0

            images.append(img)
            labels.append(category)

            # DATA AUGMENTATION
            for _ in range(AUGMENT_TIMES):
                augmented = augmentor(image=(img * 255).astype(np.uint8))
                aug_img = augmented["image"] / 255.0
                images.append(aug_img)
                labels.append(category)

        except Exception as e:
            print(f"❌ Error processing {img_path}")

print("✅ Preprocessing completed")

# ======================
# LABEL ENCODING
# ======================
images = np.array(images)
labels = np.array(labels)

encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

print("📊 Classes:", encoder.classes_)

# ======================
# TRAIN / TEST SPLIT
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    images, labels_categorical,
    test_size=0.2,
    random_state=42,
    stratify=labels_encoded
)

print("🧪 Training samples:", X_train.shape[0])
print("🧪 Testing samples:", X_test.shape[0])

# ======================
# MODEL
# ======================
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(len(encoder.classes_), activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ======================
# TRAIN
# ======================
print("🚀 Training started...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

print("✅ Training completed")

# ======================
# EVALUATE
# ======================
loss, accuracy = model.evaluate(X_test, y_test)
print(f"📈 Test Accuracy: {accuracy * 100:.2f}%")

# ======================
# VISUALIZE
# ======================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss")
plt.legend()

plt.show()

# ======================
# SAVE MODEL
# ======================
model.save("dr_eye_disease_model.h5")
print("💾 Model saved successfully")
