#!/usr/bin/env python3
"""Build the numerical differentiation report PDF with vector plots."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "numerical_differentiation_results"
OUTPUT = ROOT / "numerical_differentiation_lab_report.pdf"
X0 = 1.0
STEPS = [2.0 / (2.0**n) for n in range(1, 22)]
PALETTE = [colors.HexColor(c) for c in ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e")]

spec = importlib.util.spec_from_file_location("numerical_differentiation_lab", ROOT / "numerical_differentiation_lab.py")
lab = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = lab
spec.loader.exec_module(lab)

font_dir = Path("/usr/share/fonts/truetype/dejavu")
pdfmetrics.registerFont(TTFont("DejaVu", str(font_dir / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(font_dir / "DejaVuSans-Bold.ttf")))

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleRu", parent=styles["Title"], fontName="DejaVu-Bold", fontSize=22, leading=29, alignment=TA_CENTER, textColor=colors.HexColor("#19324a"), spaceAfter=14))
styles.add(ParagraphStyle(name="SubtitleRu", parent=styles["Normal"], fontName="DejaVu", fontSize=12, leading=18, alignment=TA_CENTER, textColor=colors.HexColor("#526477")))
styles.add(ParagraphStyle(name="HeadingRu", parent=styles["Heading2"], fontName="DejaVu-Bold", fontSize=14, leading=19, textColor=colors.HexColor("#19324a"), spaceBefore=10, spaceAfter=6))
styles.add(ParagraphStyle(name="BodyRu", parent=styles["BodyText"], fontName="DejaVu", fontSize=10, leading=15, spaceAfter=6))
styles.add(ParagraphStyle(name="SmallRu", parent=styles["BodyText"], fontName="DejaVu", fontSize=8.5, leading=12))


class ErrorPlot(Flowable):
    """Vector log-log chart with five finite-difference error curves."""

    def __init__(self, function_spec, errors):
        super().__init__()
        self.function_spec = function_spec
        self.errors = errors
        self.width = 174 * mm
        self.height = 204 * mm

    def draw(self):
        c = self.canv
        w, h = self.width, self.height
        left, right, bottom, top = 22 * mm, w - 7 * mm, 27 * mm, h - 19 * mm
        pw, ph = right - left, top - bottom
        values = [max(v, 1e-18) for series in self.errors.values() for v in series]
        x0, x1 = math.floor(math.log10(min(STEPS))) - 0.2, math.ceil(math.log10(max(STEPS))) + 0.2
        y0, y1 = math.floor(math.log10(min(values))) - 0.5, math.ceil(math.log10(max(values))) + 0.5

        def sx(value):
            return left + (math.log10(value) - x0) / (x1 - x0) * pw

        def sy(value):
            return bottom + (math.log10(max(value, 1e-18)) - y0) / (y1 - y0) * ph

        c.setFont("DejaVu-Bold", 13)
        c.setFillColor(colors.HexColor("#19324a"))
        c.drawCentredString(w / 2, h - 7 * mm, self.function_spec.name)
        c.setFont("DejaVu", 8.5)
        c.setFillColor(colors.HexColor("#526477"))
        c.drawCentredString(w / 2, h - 13 * mm, "Абсолютная погрешность при x₀ = 1")

        c.setStrokeColor(colors.HexColor("#d9e0e6"))
        c.setLineWidth(0.45)
        for exponent in range(math.ceil(x0), math.floor(x1) + 1):
            x = left + (exponent - x0) / (x1 - x0) * pw
            c.setDash(1.5, 2.5)
            c.line(x, bottom, x, top)
            c.setDash()
            label = "1" if exponent == 0 else f"10^{exponent}"
            c.setFont("DejaVu", 7.5)
            c.setFillColor(colors.HexColor("#384553"))
            c.drawCentredString(x, bottom - 5 * mm, label)
        for exponent in range(math.ceil(y0), math.floor(y1) + 1):
            y = bottom + (exponent - y0) / (y1 - y0) * ph
            c.setDash(1.5, 2.5)
            c.line(left, y, right, y)
            c.setDash()
            label = "1" if exponent == 0 else f"10^{exponent}"
            c.setFont("DejaVu", 7.5)
            c.setFillColor(colors.HexColor("#384553"))
            c.drawRightString(left - 2 * mm, y - 2.5, label)

        c.setStrokeColor(colors.HexColor("#263746"))
        c.setLineWidth(0.8)
        c.rect(left, bottom, pw, ph, stroke=1, fill=0)
        c.setFont("DejaVu", 9)
        c.setFillColor(colors.HexColor("#263746"))
        c.drawCentredString((left + right) / 2, 7 * mm, "Шаг численного дифференцирования h")
        c.saveState()
        c.translate(6 * mm, (bottom + top) / 2)
        c.rotate(90)
        c.drawCentredString(0, 0, "Абсолютная погрешность ε(h)")
        c.restoreState()

        legend_x, legend_y = right - 56 * mm, top - 7 * mm
        c.setFillColor(colors.white)
        c.setStrokeColor(colors.HexColor("#cbd4dc"))
        c.roundRect(legend_x - 3 * mm, legend_y - 30 * mm, 59 * mm, 34 * mm, 2 * mm, stroke=1, fill=1)
        for i, ((label, _), color) in enumerate(zip(lab.METHODS, PALETTE)):
            y = legend_y - i * 6 * mm
            c.setStrokeColor(color)
            c.setLineWidth(1.6)
            c.line(legend_x, y, legend_x + 7 * mm, y)
            c.setFillColor(colors.HexColor("#24313d"))
            c.setFont("DejaVu", 7)
            c.drawString(legend_x + 9 * mm, y - 2.3, label)

        for (label, _), color in zip(lab.METHODS, PALETTE):
            series = [max(value, 1e-18) for value in self.errors[label]]
            c.setStrokeColor(color)
            c.setFillColor(color)
            c.setLineWidth(1.35)
            path = c.beginPath()
            path.moveTo(sx(STEPS[0]), sy(series[0]))
            for step, value in zip(STEPS[1:], series[1:]):
                path.lineTo(sx(step), sy(value))
            c.drawPath(path, stroke=1, fill=0)
            for step, value in zip(STEPS, series):
                c.circle(sx(step), sy(value), 1.45, stroke=0, fill=1)


def footer(canvas, doc):
    canvas.saveState()
    page_w, _ = A4
    canvas.setStrokeColor(colors.HexColor("#d8e0e7"))
    canvas.setFont("DejaVu", 8)
    canvas.setFillColor(colors.HexColor("#6a7885"))
    canvas.drawRightString(page_w - 20 * mm, 10 * mm, str(doc.page))
    canvas.restoreState()


def make_story():
    story = [Spacer(1, 12 * mm), Paragraph("Лабораторная работа", styles["TitleRu"]), Paragraph("Численное дифференцирование", styles["TitleRu"]), Spacer(1, 3 * mm), Paragraph("Исследование абсолютной погрешности пяти разностных формул", styles["SubtitleRu"]), Spacer(1, 8 * mm), Paragraph("Авторы: Дернович Арсений, Русецкий Егор", styles["SubtitleRu"]), Spacer(1, 10 * mm)]
    story.append(Paragraph("Цель работы", styles["HeadingRu"]))
    story.append(Paragraph("Исследовать зависимость абсолютной погрешности численного дифференцирования от шага h и сравнить точность пяти разностных формул на заданных функциях.", styles["BodyRu"]))
    story.append(Paragraph("Исходные данные", styles["HeadingRu"]))
    story.append(Paragraph("Шаг изменяется по закону hₙ = 2 / 2ⁿ, n = 1,…,21. В предоставленной фотографии точка вычисления не указана, поэтому в работе принято x₀ = 1. Для каждой функции построен отдельный график в логарифмическом масштабе по обеим осям.", styles["BodyRu"]))
    story.append(Paragraph("Исследуемые функции", styles["HeadingRu"]))
    functions_text = "sin(x²); &nbsp; cos(sin x); &nbsp; exp(sin(cos x)); &nbsp; ln(x + 3); &nbsp; √(x + 3)."
    story.append(Paragraph(functions_text, styles["BodyRu"]))
    story.append(Paragraph("Формулы численного дифференцирования", styles["HeadingRu"]))
    formulas = [
        ["1", "Прямая разность", "[f(x+h) − f(x)] / h"],
        ["2", "Обратная разность", "[f(x) − f(x−h)] / h"],
        ["3", "Центральная разность", "[f(x+h) − f(x−h)] / (2h)"],
        ["4", "Симметричная формула", "(4/3)[f(x+h)−f(x−h)]/(2h) − (1/3)[f(x+2h)−f(x−2h)]/(4h)"],
        ["5", "Симметричная формула", "(3/2)[f(x+h)−f(x−h)]/(2h) − (3/5)[f(x+2h)−f(x−2h)]/(4h) + (1/10)[f(x+3h)−f(x−3h)]/(6h)"],
    ]
    table = Table(formulas, colWidths=[9 * mm, 43 * mm, 119 * mm], repeatRows=0)
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "DejaVu"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.4),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#263746")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f3f6f8")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#f3f6f8"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d8e0e7")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)
    story.append(Paragraph("Аналитические производные", styles["HeadingRu"]))
    derivatives = [
        "sin(x²): &nbsp; 2x cos(x²)",
        "cos(sin x): &nbsp; −sin(sin x) cos x",
        "exp(sin(cos x)): &nbsp; −exp(sin(cos x)) cos(cos x) sin x",
        "ln(x + 3): &nbsp; 1/(x + 3)",
        "√(x + 3): &nbsp; 1/(2√(x + 3))",
    ]
    for formula in derivatives:
        story.append(Paragraph("• &nbsp;" + formula, styles["SmallRu"]))
    story.append(Paragraph("Абсолютная погрешность", styles["HeadingRu"]))
    story.append(Paragraph("Для каждого шага вычислялось ε(h) = |Dₕf(x₀) − f′(x₀)|, где Dₕ — соответствующая разностная формула, а f′(x₀) — аналитическая производная.", styles["BodyRu"]))

    for function_spec in lab.FUNCTIONS:
        story.append(PageBreak())
        errors = lab.compute_errors(function_spec, X0, STEPS)
        story.append(ErrorPlot(function_spec, errors))

    return story


def main():
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=20 * mm, leftMargin=20 * mm, topMargin=19 * mm, bottomMargin=21 * mm, title="Лабораторная работа: численное дифференцирование", author="Дернович Арсений; Русецкий Егор")
    doc.build(make_story(), onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    main()
