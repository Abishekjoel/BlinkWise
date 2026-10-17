"""PDF report generation for a single Diagnosis record."""
import io

from django.utils import timezone
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, PageTemplate, Paragraph,
    Spacer, Table, TableStyle,
)
from reportlab.graphics.shapes import Drawing, Rect

from .ml_model import SEVERITY_INFO

NAVY = colors.HexColor("#0b1730")
BLUE = colors.HexColor("#4f7cff")
LIGHT = colors.HexColor("#eef2ff")
GRID = colors.HexColor("#d5dbee")
TEXT = colors.HexColor("#1b2338")
MUTED = colors.HexColor("#6b7694")

SEVERITY_COLORS = {
    "No_DR": colors.HexColor("#16a34a"),
    "Mild": colors.HexColor("#65a30d"),
    "Moderate": colors.HexColor("#d97706"),
    "Severe": colors.HexColor("#ea580c"),
    "Proliferate_DR": colors.HexColor("#dc2626"),
}

# General screening follow-up guidance per stage (informational only).
FOLLOW_UP = {
    "No_DR": (
        "No signs of diabetic retinopathy were detected.",
        "Continue good glycaemic, blood pressure and lipid control. Routine retinal screening in about 12 months.",
    ),
    "Mild": (
        "Early changes consistent with mild non-proliferative retinopathy (e.g. microaneurysms).",
        "Optimise blood sugar and blood pressure control. Repeat retinal screening in 6 to 12 months.",
    ),
    "Moderate": (
        "More widespread vascular changes consistent with moderate non-proliferative retinopathy.",
        "Referral to an ophthalmologist is advised. Repeat examination within 3 to 6 months, with tight systemic control.",
    ),
    "Severe": (
        "Extensive vascular damage with a high risk of progression to proliferative disease.",
        "Prompt referral to an ophthalmologist or retina specialist is recommended (within weeks).",
    ),
    "Proliferate_DR": (
        "Advanced disease with abnormal new vessel growth that can threaten vision.",
        "Urgent referral to a retina specialist is recommended. Treatment may be required to prevent vision loss.",
    ),
}


def _styles():
    base = getSampleStyleSheet()
    return {
        "body": ParagraphStyle("body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.5, leading=14, textColor=TEXT),
        "small": ParagraphStyle("small", parent=base["BodyText"], fontName="Helvetica", fontSize=8, leading=11, textColor=MUTED),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12, textColor=NAVY, spaceBefore=14, spaceAfter=6),
        "label": ParagraphStyle("label", fontName="Helvetica-Bold", fontSize=7.5, textColor=MUTED, leading=10),
        "value": ParagraphStyle("value", fontName="Helvetica-Bold", fontSize=10.5, textColor=TEXT, leading=14),
        "cap": ParagraphStyle("cap", fontName="Helvetica", fontSize=8, textColor=MUTED, alignment=TA_CENTER, leading=11),
    }


def _section(title, st):
    t = Table([[Paragraph(title.upper(), st["h2"])]], colWidths=[174 * mm])
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 1.2, BLUE),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
    ]))
    return t


def _bar(pct, color, width=70 * mm, height=4 * mm):
    d = Drawing(width, height)
    d.add(Rect(0, 0, width, height, fillColor=colors.HexColor("#e6eaf7"), strokeColor=None, rx=2, ry=2))
    fill = max(0.0, min(100.0, pct)) / 100.0 * width
    if fill > 0:
        d.add(Rect(0, 0, fill, height, fillColor=color, strokeColor=None, rx=2, ry=2))
    return d


def _page_decor(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, h - 28 * mm, w, 28 * mm, stroke=0, fill=1)
    canvas.setFillColor(BLUE)
    canvas.rect(0, h - 29.2 * mm, w, 1.2 * mm, stroke=0, fill=1)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 19)
    canvas.drawString(18 * mm, h - 15 * mm, "BlinkWise")
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#a9b8e6"))
    canvas.drawString(18 * mm, h - 21 * mm, "See beyond the blink")
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 12)
    canvas.drawRightString(w - 18 * mm, h - 14 * mm, "Diabetic Retinopathy Screening Report")
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(colors.HexColor("#a9b8e6"))
    canvas.drawRightString(w - 18 * mm, h - 20 * mm, f"Report ID: {doc.report_id}")

    canvas.setStrokeColor(GRID)
    canvas.line(18 * mm, 16 * mm, w - 18 * mm, 16 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 11.5 * mm, "BlinkWise AI screening tool. Not a medical diagnosis.")
    canvas.drawRightString(w - 18 * mm, 11.5 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _image(field, size):
    try:
        img = Image(field.path, width=size, height=size)
        img.hAlign = "CENTER"
        return img
    except Exception:
        return Paragraph("Image not available", _styles()["cap"])


def build_report_pdf(record):
    st = _styles()
    info = SEVERITY_INFO.get(record.predicted_label, {})
    sev_color = SEVERITY_COLORS.get(record.predicted_label, BLUE)
    finding, advice = FOLLOW_UP.get(record.predicted_label, ("", ""))
    gender = record.get_patient_gender_display() if record.patient_gender else "Not specified"
    created = timezone.localtime(record.created_at)

    buf = io.BytesIO()
    doc = BaseDocTemplate(
        buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=36 * mm, bottomMargin=22 * mm,
        title=f"BlinkWise Report {record.patient_code}", author="BlinkWise",
    )
    doc.report_id = f"{record.patient_code}-{created:%Y%m%d}"
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=_page_decor)])

    story = []

    # --- Patient information
    story.append(_section("Patient information", st))
    story.append(Spacer(1, 4 * mm))

    def cell(label, value):
        return [Paragraph(label.upper(), st["label"]), Paragraph(str(value), st["value"])]

    info_rows = [
        [cell("Patient name", record.patient_name), cell("Patient ID", record.patient_code), cell("Age", f"{record.patient_age} years")],
        [cell("Gender", gender), cell("Scan date", f"{created:%d %b %Y, %I:%M %p}"), cell("Report generated", f"{timezone.localtime():%d %b %Y, %I:%M %p}")],
    ]
    t = Table(info_rows, colWidths=[58 * mm] * 3)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.6, GRID),
        ("INNERGRID", (0, 0), (-1, -1), 0.6, colors.white),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(t)

    # --- Diagnosis result
    story.append(_section("Diagnosis result", st))
    story.append(Spacer(1, 4 * mm))
    result_tbl = Table([[
        [Paragraph("PRIMARY FINDING", ParagraphStyle("l", parent=st["label"], textColor=colors.white)),
         Paragraph(record.predicted_label.replace("_", " "), ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=20, textColor=colors.white, leading=24)),
         Paragraph(info.get("clinical_name", ""), ParagraphStyle("s", fontName="Helvetica", fontSize=9.5, textColor=colors.white, leading=13))],
        [Paragraph("MODEL CONFIDENCE", ParagraphStyle("l2", parent=st["label"], textColor=colors.white, alignment=TA_CENTER)),
         Paragraph(f"{record.confidence:.1f}%", ParagraphStyle("c", fontName="Helvetica-Bold", fontSize=24, textColor=colors.white, leading=30, alignment=TA_CENTER)),
         Paragraph(f"Stage {info.get('rank', 0) + 1} of 5", ParagraphStyle("st", fontName="Helvetica", fontSize=9, textColor=colors.white, alignment=TA_CENTER))],
    ]], colWidths=[110 * mm, 64 * mm])
    result_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), sev_color),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 14), ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 12), ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LINEAFTER", (0, 0), (0, 0), 0.6, colors.Color(1, 1, 1, 0.5)),
    ]))
    story.append(result_tbl)

    # --- Images
    story.append(_section("Retinal imaging", st))
    story.append(Spacer(1, 4 * mm))
    size = 78 * mm
    pics = [[_image(record.image, size), _image(record.heatmap, size) if record.heatmap else Paragraph("Heatmap not available", st["cap"])],
            [Paragraph("Original fundus image", st["cap"]), Paragraph("AI heatmap (Grad-CAM): warmer areas influenced the prediction most", st["cap"])]]
    pic_tbl = Table(pics, colWidths=[87 * mm, 87 * mm])
    pic_tbl.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 1), (-1, 1), 4),
    ]))
    story.append(pic_tbl)

    # --- Confidence breakdown
    probs = sorted(record.probabilities or [], key=lambda p: p.get("rank", 0))
    prob_rows = [[Paragraph("STAGE", st["label"]), Paragraph("CLASSIFICATION", st["label"]), Paragraph("CONFIDENCE", st["label"]), ""]]
    for p in probs:
        label = p.get("label", "")
        is_top = label == record.predicted_label
        name_style = ParagraphStyle("n", parent=st["body"], fontName="Helvetica-Bold" if is_top else "Helvetica")
        prob_rows.append([
            Paragraph(str(p.get("rank", 0) + 1), name_style),
            Paragraph(p.get("clinical_name") or label, name_style),
            _bar(p.get("confidence", 0), SEVERITY_COLORS.get(label, BLUE)),
            Paragraph(f"{p.get('confidence', 0):.1f}%", ParagraphStyle("pc", parent=name_style, alignment=2)),
        ])
    prob_tbl = Table(prob_rows, colWidths=[14 * mm, 62 * mm, 76 * mm, 22 * mm])
    top_row = [i for i, p in enumerate(probs, start=1) if p.get("label") == record.predicted_label]
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for r in top_row:
        style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f4f7ff")))
    prob_tbl.setStyle(TableStyle(style))
    story.append(KeepTogether([_section("Confidence by stage", st), Spacer(1, 4 * mm), prob_tbl]))

    # --- Interpretation
    interp = [
        _section("Interpretation and suggested follow-up", st), Spacer(1, 3 * mm),
        Paragraph(f"<b>Finding:</b> {info.get('description', '')} {finding}", st["body"]),
        Spacer(1, 2 * mm),
        Paragraph(f"<b>Suggested follow-up:</b> {advice}", st["body"]),
    ]
    story.append(KeepTogether(interp))

    # --- Technical details
    tech_rows = [
        [cell("Analysis model", "MobileNetV2 CNN (transfer learning)"), cell("Input size", "224 x 224 px"), cell("Processing time", f"{record.processing_time_ms:.0f} ms" if record.processing_time_ms else "n/a")],
    ]
    tech = Table(tech_rows, colWidths=[70 * mm, 50 * mm, 54 * mm])
    tech.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.6, GRID), ("INNERGRID", (0, 0), (-1, -1), 0.6, GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(KeepTogether([_section("Technical details", st), Spacer(1, 4 * mm), tech]))

    # --- Disclaimer
    story.append(Spacer(1, 8 * mm))
    disc = Table([[Paragraph(
        "<b>Disclaimer.</b> This report was generated automatically by an AI screening tool for research and "
        "educational use. It is not a medical diagnosis and must not replace examination by a qualified "
        "ophthalmologist. Model accuracy varies between stages, and the more advanced stages in particular can be "
        "mistaken for one another. Please consult a healthcare professional to confirm any finding.", st["small"])]],
        colWidths=[174 * mm])
    disc.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff7ed")),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#fdba74")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(disc)

    doc.build(story)
    return buf.getvalue()
