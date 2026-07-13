import csv

from django.core.files.base import ContentFile
from django.core.paginator import Paginator
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, render

from .ml_model import SEVERITY_INFO, predict_with_heatmap
from .models import CONDITION_CHOICES, Diagnosis

PAGE_SIZE = 10


def index(request):
    if request.method == "POST" and request.FILES.get("image"):
        upload = request.FILES["image"]
        patient_name = request.POST.get("patient_name", "").strip()
        age_raw = request.POST.get("patient_age", "").strip()

        if not patient_name or not age_raw.isdigit():
            return render(request, "detector/index.html", {
                "error": "Please enter a valid patient name and age.",
                "active_page": "new_analysis",
            })

        diagnosis = Diagnosis(
            patient_name=patient_name,
            patient_age=int(age_raw),
            patient_gender=request.POST.get("patient_gender", ""),
            image=upload,
        )
        diagnosis.save()

        try:
            result, heatmap_png, processing_time_ms = predict_with_heatmap(diagnosis.image.path)
        except Exception as exc:
            diagnosis.delete()
            return render(request, "detector/index.html", {
                "error": str(exc),
                "active_page": "new_analysis",
            })

        diagnosis.predicted_label = result["label"]
        diagnosis.confidence = result["confidence"]
        diagnosis.probabilities = result["probabilities"]
        diagnosis.processing_time_ms = processing_time_ms
        diagnosis.heatmap.save(
            f"heatmap_{diagnosis.id}.png", ContentFile(heatmap_png), save=False
        )
        diagnosis.save()

        return render(request, "detector/index.html", {
            "result": result,
            "diagnosis": diagnosis,
            "active_page": "new_analysis",
        })

    return render(request, "detector/index.html", {"active_page": "new_analysis"})


def _filtered_records(request):
    queryset = Diagnosis.objects.all()

    name_query = request.GET.get("name", "").strip()
    min_age_query = request.GET.get("min_age", "").strip()
    condition_query = request.GET.get("condition", "").strip()

    if name_query:
        queryset = queryset.filter(patient_name__icontains=name_query)
    if min_age_query.isdigit():
        queryset = queryset.filter(patient_age__gte=min_age_query)
    if condition_query:
        queryset = queryset.filter(predicted_label=condition_query)

    return queryset, name_query, min_age_query, condition_query


def records(request):
    queryset, name_query, min_age_query, condition_query = _filtered_records(request)

    paginator = Paginator(queryset, PAGE_SIZE)
    page_obj = paginator.get_page(request.GET.get("page"))

    return render(request, "detector/records.html", {
        "page_obj": page_obj,
        "condition_choices": CONDITION_CHOICES,
        "name_query": name_query,
        "min_age_query": min_age_query,
        "condition_query": condition_query,
        "active_page": "records",
    })


def records_export(request):
    queryset, *_ = _filtered_records(request)

    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="patient_records.csv"'

    writer = csv.writer(response)
    writer.writerow(["Patient ID", "Name", "Age", "Gender", "Condition", "Confidence %", "Analyzed"])
    for record in queryset:
        writer.writerow([
            record.patient_code,
            record.patient_name,
            record.patient_age,
            record.get_patient_gender_display(),
            record.predicted_label,
            f"{record.confidence:.1f}",
            record.created_at.strftime("%Y-%m-%d %H:%M"),
        ])
    return response


def record_detail(request, pk):
    record = get_object_or_404(Diagnosis, pk=pk)
    info = SEVERITY_INFO.get(record.predicted_label, {})
    return render(request, "detector/record_detail.html", {
        "record": record,
        "description": info.get("description", ""),
        "clinical_name": info.get("clinical_name", ""),
        "stage_rank": info.get("rank", 0) + 1,
        "active_page": "records",
    })


def settings_page(request):
    return render(request, "detector/settings.html", {"active_page": "settings"})
