from django.db import models

GENDER_CHOICES = [
    ("M", "Male"),
    ("F", "Female"),
    ("O", "Other"),
]

CONDITION_CHOICES = [
    ("No_DR", "No_DR"),
    ("Mild", "Mild"),
    ("Moderate", "Moderate"),
    ("Severe", "Severe"),
    ("Proliferate_DR", "Proliferate_DR"),
]


class Diagnosis(models.Model):
    patient_name = models.CharField(max_length=200)
    patient_age = models.PositiveIntegerField()
    patient_gender = models.CharField(max_length=1, choices=GENDER_CHOICES, blank=True)

    image = models.ImageField(upload_to="uploads/")
    heatmap = models.ImageField(upload_to="heatmaps/", blank=True, null=True)

    predicted_label = models.CharField(max_length=30, choices=CONDITION_CHOICES, blank=True)
    confidence = models.FloatField(default=0.0)
    probabilities = models.JSONField(default=list, blank=True)
    processing_time_ms = models.FloatField(default=0.0)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.patient_name} ({self.created_at:%Y-%m-%d %H:%M}) - {self.predicted_label}"

    @property
    def patient_code(self):
        """Display-only identifier derived from the primary key (not a stored/fabricated ID)."""
        return f"PID-{self.pk:05d}"
