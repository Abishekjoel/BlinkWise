from django.contrib import admin

from .models import Diagnosis


@admin.register(Diagnosis)
class DiagnosisAdmin(admin.ModelAdmin):
    list_display = ("patient_name", "patient_age", "patient_gender", "predicted_label", "confidence", "created_at")
    list_filter = ("predicted_label", "patient_gender")
    search_fields = ("patient_name",)
