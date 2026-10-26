#!/usr/bin/env python3
"""Third numerical methods lab: scalar equations and nonlinear systems.

Selected exercise variants are listed in PROBLEMS. Results are saved as JSON,
CSV, SVG convergence charts, and a compact PDF report.

Dependencies: numpy and reportlab.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


TOL = 1.0e-10
MAX_ITER = 10000
PALETTE = ["#2364aa", "#d1495b", "#2a9d8f", "#8a5fbf", "#ed8b00"]


@dataclass(frozen=True)
class Problem:
    key: str
    title: str
    source: str
    kind: str
    formula: str
    function: Callable
    jacobian: Callable | None
    fixed_map: Callable
    initial: np.ndarray
    bracket: tuple[float, float] | None = None
    alpha_fixed: float | None = None
    fixed_formula: str = ""
    start_note: str = ""
    isolation_note: str = ""


PROBLEMS = [
    Problem(
        key="iv12_3b",
        title="IV.12.3, вариант б)",
        source="Уравнение 4x³ − 12x² + 3x − 5 = 0",
        kind="scalar",
        formula="F(x) = 4x³ − 12x² + 3x − 5",
        function=lambda x: 4.0 * x**3 - 12.0 * x**2 + 3.0 * x - 5.0,
        jacobian=lambda x: 12.0 * x**2 - 24.0 * x + 3.0,
        fixed_map=lambda x: x - 0.03 * (4.0 * x**3 - 12.0 * x**2 + 3.0 * x - 5.0),
        initial=np.array([2.9]),
        bracket=(2.8, 3.0),
        alpha_fixed=0.03,
        fixed_formula="φ(x)=x−0,03F(x)",
        start_note="отрезок [2,8; 3], x₀=2,9",
        isolation_note="У F' нули при x=1±√3/2, в обеих точках F<0; поскольку F(x)→+∞ при x→+∞, единственный корень лежит на [2,8; 3].",
    ),
    Problem(
        key="iv12_3d",
        title="IV.12.3, вариант д)",
        source="Уравнение arctg(x − 1) + 2x = 0",
        kind="scalar",
        formula="F(x) = arctan(x − 1) + 2x",
        function=lambda x: math.atan(x - 1.0) + 2.0 * x,
        jacobian=lambda x: 1.0 / (1.0 + (x - 1.0) ** 2) + 2.0,
        fixed_map=lambda x: -0.5 * math.atan(x - 1.0),
        initial=np.array([0.5]),
        bracket=(0.0, 1.0),
        fixed_formula="φ(x)=−0,5 arctan(x−1)",
        start_note="отрезок [0; 1], x₀=0,5",
        isolation_note="F'(x)=2+1/(1+(x−1)²)>0, поэтому корень единственный; F(0)<0<F(1).",
    ),
    Problem(
        key="iv12_4a",
        title="IV.12.4, вариант а)",
        source="Пересечение sin(x + 1) − y = 1,2;  2x + cos y = 2",
        kind="system",
        formula="F₁ = sin(x + 1) − y − 1,2;   F₂ = 2x + cos y − 2",
        function=lambda u: np.array([math.sin(u[0] + 1.0) - u[1] - 1.2, 2.0 * u[0] + math.cos(u[1]) - 2.0]),
        jacobian=lambda u: np.array([[math.cos(u[0] + 1.0), -1.0], [2.0, -math.sin(u[1])]]),
        fixed_map=lambda u: np.array([1.0 - 0.5 * math.cos(u[1]), math.sin(u[0] + 1.0) - 1.2]),
        initial=np.array([1.0, -1.0]),
        fixed_formula="xₖ₊₁=1−0,5cos yₖ;  yₖ₊₁=sin(xₖ+1)−1,2",
        start_note="начальное приближение (x₀,y₀)=(1;−1)",
        isolation_note="Эквивалентное отображение x=1−0,5cos y, y=sin(x+1)−1,2 сжимает прямоугольник [0,5;1,5]×[−2,2;−0,2]; в нём пересечение единственно.",
    ),
    Problem(
        key="iv12_6g",
        title="IV.12.6, вариант г)",
        source="Пересечение sin x − y = 1,32;  cos y − x = −0,85",
        kind="system",
        formula="F₁ = sin x − y − 1,32;   F₂ = cos y − x + 0,85",
        function=lambda u: np.array([math.sin(u[0]) - u[1] - 1.32, math.cos(u[1]) - u[0] + 0.85]),
        jacobian=lambda u: np.array([[math.cos(u[0]), -1.0], [-1.0, -math.sin(u[1])]]),
        fixed_map=lambda u: np.array([math.cos(u[1]) + 0.85, math.sin(u[0]) - 1.32]),
        initial=np.array([1.6, -0.4]),
        fixed_formula="xₖ₊₁=cos yₖ+0,85;  yₖ₊₁=sin xₖ−1,32",
        start_note="начальное приближение (x₀,y₀)=(1,6;−0,4)",
        isolation_note="Из первого уравнения y=sin x−1,32, из второго x=cos y+0,85, поэтому x∈[−0,15;1,85]. На этом интервале приведённая функция cos(sin x−1,32)−x+0,85 строго убывает и имеет единственный нуль.",
    ),
]


def norm_f(problem: Problem, x: np.ndarray) -> float:
    val = np.asarray(problem.function(float(x[0]) if problem.kind == "scalar" else x), dtype=float)
    return float(np.linalg.norm(val))


def bisect(problem: Problem, tol=TOL):
    a, b = problem.bracket
    fa, fb = float(problem.function(a)), float(problem.function(b))
    if fa * fb > 0:
        raise ValueError("На отрезке не найдено смены знака")
    history = []
    trajectory = []
    for k in range(1, MAX_ITER + 1):
        m = (a + b) / 2.0
        fm = float(problem.function(m))
        history.append(abs(fm))
        trajectory.append(m)
        if abs(fm) <= tol or m == a or m == b:
            return np.array([m]), history, trajectory, "сошёлся"
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return np.array([(a + b) / 2.0]), history, trajectory, "достигнут лимит"


def fixed_point(problem: Problem, tol=TOL):
    x = problem.initial.astype(float).copy()
    history = [norm_f(problem, x)]
    trajectory = [x.copy()]
    for _ in range(MAX_ITER):
        if problem.kind == "scalar":
            xn = np.array([problem.fixed_map(float(x[0]))])
        else:
            xn = np.asarray(problem.fixed_map(x), dtype=float)
        if not np.all(np.isfinite(xn)):
            return x, history, trajectory, "расходимость: нечисловое значение"
        x = xn
        history.append(norm_f(problem, x))
        trajectory.append(x.copy())
        if history[-1] <= tol:
            return x, history, trajectory, "сошёлся"
    return x, history, trajectory, "достигнут лимит"


def newton_scalar(problem: Problem, modified=False, tol=TOL):
    x = float(problem.initial[0])
    derivative0 = float(problem.jacobian(x))
    if abs(derivative0) < 1e-15:
        raise ValueError("Производная близка к нулю в начальной точке")
    history = [abs(float(problem.function(x)))]
    trajectory = [x]
    for _ in range(MAX_ITER):
        fx = float(problem.function(x))
        derivative = derivative0 if modified else float(problem.jacobian(x))
        if abs(derivative) < 1e-15:
            return np.array([x]), history, trajectory, "остановлен: производная близка к нулю"
        xn = x - fx / derivative
        if not math.isfinite(xn):
            return np.array([x]), history, trajectory, "расходимость: нечисловое значение"
        x = xn
        trajectory.append(x)
        history.append(abs(float(problem.function(x))))
        if history[-1] <= tol:
            return np.array([x]), history, trajectory, "сошёлся"
    return np.array([x]), history, trajectory, "достигнут лимит"


def newton_system(problem: Problem, modified=False, tol=TOL):
    x = problem.initial.astype(float).copy()
    J0 = np.asarray(problem.jacobian(x), dtype=float) if modified else None
    history = [norm_f(problem, x)]
    trajectory = [x.copy()]
    for _ in range(MAX_ITER):
        J = J0 if modified else np.asarray(problem.jacobian(x), dtype=float)
        try:
            delta = np.linalg.solve(J, np.asarray(problem.function(x), dtype=float))
        except np.linalg.LinAlgError:
            return x, history, trajectory, "остановлен: вырожденный якобиан"
        # Backtracking protects the local Newton method from a bad global step.
        step = 1.0
        phi = 0.5 * history[-1] ** 2
        while step > 2.0**-24:
            candidate = x - step * delta
            cand_norm = norm_f(problem, candidate)
            if 0.5 * cand_norm**2 <= phi * (1.0 - 1e-4 * step):
                break
            step *= 0.5
        if step <= 2.0**-24:
            return x, history, trajectory, "остановлен: шаг Ньютона не уменьшает невязку"
        x = candidate
        trajectory.append(x.copy())
        history.append(cand_norm)
        if history[-1] <= tol:
            return x, history, trajectory, "сошёлся"
    return x, history, trajectory, "достигнут лимит"


def variational_steepest_descent(problem: Problem, tol=TOL):
    """Minimize Phi(x)=1/2 ||F(x)||^2 by steepest descent with Armijo search."""
    x = problem.initial.astype(float).copy()
    history = [norm_f(problem, x)]
    trajectory = [x.copy()]
    for _ in range(MAX_ITER):
        f = np.atleast_1d(np.asarray(problem.function(float(x[0]) if problem.kind == "scalar" else x), dtype=float))
        if problem.kind == "scalar":
            J = np.array([[float(problem.jacobian(float(x[0])))]])
        else:
            J = np.asarray(problem.jacobian(x), dtype=float)
        grad = J.T @ f  # gradient of 1/2 ||F||^2
        grad_norm2 = float(grad @ grad)
        if grad_norm2 <= np.finfo(float).tiny:
            return x, history, trajectory, "остановлен: градиент функционала близок к нулю"
        phi = 0.5 * float(f @ f)
        alpha = 1.0
        while alpha > 2.0**-30:
            candidate = x - alpha * grad
            candidate_f = np.atleast_1d(np.asarray(problem.function(float(candidate[0]) if problem.kind == "scalar" else candidate), dtype=float))
            candidate_phi = 0.5 * float(candidate_f @ candidate_f)
            if candidate_phi <= phi - 1e-4 * alpha * grad_norm2:
                break
            alpha *= 0.5
        if alpha <= 2.0**-30:
            return x, history, trajectory, "остановлен: линейный поиск не уменьшает функционал"
        x = candidate
        trajectory.append(x.copy())
        history.append(float(np.linalg.norm(candidate_f)))
        if history[-1] <= tol:
            return x, history, trajectory, "сошёлся"
    return x, history, trajectory, "достигнут лимит"


def run_problem(problem: Problem):
    results = {}
    methods = []
    if problem.kind == "scalar":
        methods.append(("Деление пополам", lambda: bisect(problem)))
        methods.append(("Простая итерация", lambda: fixed_point(problem)))
        methods.append(("Ньютон", lambda: newton_scalar(problem)))
        methods.append(("Модифицированный Ньютон", lambda: newton_scalar(problem, modified=True)))
    else:
        methods.append(("Простая итерация", lambda: fixed_point(problem)))
        methods.append(("Ньютон", lambda: newton_system(problem)))
        methods.append(("Модифицированный Ньютон", lambda: newton_system(problem, modified=True)))
    methods.append(("Вариационный спуск", lambda: variational_steepest_descent(problem)))
    for name, solver in methods:
        root, history, trajectory, status = solver()
        results[name] = {
            "root": np.asarray(root, dtype=float).tolist(),
            "history": [float(v) for v in history],
            "trajectory": np.asarray(trajectory, dtype=float).tolist(),
            "iterations": max(0, len(history) - 1),
            "residual": float(history[-1]),
            "status": status,
        }
    return results


def svg_chart(problem: Problem, results: dict[str, dict], path: Path):
    width, height = 1050, 620
    left, right, top, bottom = 105, 720, 55, 515
    histories = [(name, data["history"]) for name, data in results.items()]
    values = [max(v, 1e-30) for _, seq in histories for v in seq]
    ymin, ymax = math.floor(math.log10(min(values))), math.ceil(math.log10(max(values)))
    if ymin == ymax:
        ymin -= 1
        ymax += 1
    max_iter = max(len(seq) - 1 for _, seq in histories)

    def sx(k):
        return left + (right - left) * k / max(1, max_iter)

    def sy(v):
        return bottom - (bottom - top) * (math.log10(max(v, 1e-30)) - ymin) / (ymax - ymin)

    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{width/2}" y="30" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="20" fill="#19324a">Сходимость методов: {problem.title}</text>']
    for p in range(ymin, ymax + 1):
        y = bottom - (bottom - top) * (p - ymin) / (ymax - ymin)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#dce2e7"/>')
        lines.append(f'<text x="{left-12}" y="{y+4:.2f}" text-anchor="end" font-family="DejaVu Sans,sans-serif" font-size="12" fill="#465563">10^{p}</text>')
    xticks = sorted(set([0, max_iter // 4, max_iter // 2, (3 * max_iter) // 4, max_iter]))
    for k in xticks:
        x = sx(k)
        lines.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" stroke="#edf0f2"/>')
        lines.append(f'<text x="{x:.2f}" y="{bottom+22}" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="12" fill="#465563">{k}</text>')
    lines.extend([f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="none" stroke="#33495c"/>', f'<text x="{(left+right)/2}" y="{bottom+50}" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="14" fill="#243746">Номер итерации k</text>', f'<text x="25" y="{(top+bottom)/2}" transform="rotate(-90 25 {(top+bottom)/2})" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="14" fill="#243746">Норма невязки ||F(xₖ)||₂</text>'])
    for index, (name, seq) in enumerate(histories):
        color = PALETTE[index % len(PALETTE)]
        d = " ".join(("M" if i == 0 else "L") + f"{sx(i):.2f},{sy(v):.2f}" for i, v in enumerate(seq))
        lines.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        for i, value in enumerate(seq):
            lines.append(f'<circle cx="{sx(i):.2f}" cy="{sy(value):.2f}" r="3" fill="{color}"/>')
        ly = 100 + index * 42
        lines.append(f'<line x1="760" y1="{ly}" x2="792" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="805" y="{ly+5}" font-family="DejaVu Sans,sans-serif" font-size="14" fill="#243746">{name}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def export_csv(directory: Path, problem_results: dict[str, dict]):
    path = directory / "iteration_history.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["problem", "method", "iteration", "residual"])
        for key, results in problem_results.items():
            for method, item in results.items():
                for k, residual in enumerate(item["history"]):
                    writer.writerow([key, method, k, f"{residual:.16e}"])


def draw_wrapped(c, text: str, x: float, y: float, max_width: float, font="DV", size=8.5, leading=11):
    c.setFont(font, size)
    words = text.split()
    line = ""
    for word in words:
        trial = (line + " " + word).strip()
        if pdfmetrics.stringWidth(trial, font, size) > max_width and line:
            c.drawString(x, y, line)
            y -= leading
            line = word
        else:
            line = trial
    if line:
        c.drawString(x, y, line)
        y -= leading
    return y


def create_pdf(path: Path, problem_results: dict[str, dict]):
    regular = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    bold = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    pdfmetrics.registerFont(TTFont("DV", str(regular)))
    pdfmetrics.registerFont(TTFont("DVB", str(bold)))
    c = canvas.Canvas(str(path), pagesize=A4)
    w, h = A4
    navy = colors.HexColor("#19324a")
    muted = colors.HexColor("#5b6b78")
    c.setTitle("Третья лабораторная: нелинейные уравнения и системы")
    c.setAuthor("Дернович Арсений; Русецкий Егор")

    # Overview page.
    c.setFillColor(navy)
    c.setFont("DVB", 18)
    c.drawCentredString(w / 2, h - 24 * mm, "Лабораторная работа № 3")
    c.setFont("DVB", 14)
    c.drawCentredString(w / 2, h - 33 * mm, "Нелинейные уравнения и системы")
    c.setFont("DV", 10)
    c.setFillColor(muted)
    c.drawCentredString(w / 2, h - 43 * mm, "Авторы: Дернович Арсений, Русецкий Егор")
    c.showPage()

    # One page per task, with method summary and a log-residual graph.
    for p in PROBLEMS:
        rr = problem_results[p.key]
        c.setFillColor(navy)
        c.setFont("DVB", 15)
        c.drawCentredString(w / 2, h - 20 * mm, p.title)
        c.setFont("DV", 9)
        c.setFillColor(muted)
        c.drawCentredString(w / 2, h - 28 * mm, p.source)
        c.setFillColor(navy)
        c.setFont("DVB", 9.5)
        c.drawString(20 * mm, h - 41 * mm, "Система/функция и результат")
        c.setFillColor(colors.HexColor("#263746"))
        root = next(iter(rr.values()))["root"]
        result_text = f"{p.formula};  решение: x = {root[0]:.12g}" if p.kind == "scalar" else f"{p.formula};  решение: (x,y)=({root[0]:.10g}, {root[1]:.10g})"
        text_y = draw_wrapped(c, result_text, 22 * mm, h - 48 * mm, w - 44 * mm, size=8.5, leading=10)
        iteration_text = f"Простая итерация: {p.fixed_formula}; {p.start_note}."
        text_y = draw_wrapped(c, iteration_text, 22 * mm, text_y - 1.5 * mm, w - 44 * mm, size=7.8, leading=9)
        text_y = draw_wrapped(c, p.isolation_note, 22 * mm, text_y - 1 * mm, w - 44 * mm, size=7.4, leading=8.5)

        table_top = text_y - 4 * mm
        c.setFillColor(colors.HexColor("#eaf0f4"))
        c.rect(20 * mm, table_top - 4 * mm, w - 40 * mm, 7 * mm, fill=1, stroke=0)
        c.setFillColor(navy)
        c.setFont("DVB", 7.7)
        c.drawString(22 * mm, table_top - 1.5 * mm, "Метод")
        c.drawRightString(w - 55 * mm, table_top - 1.5 * mm, "Итерации")
        c.drawRightString(w - 22 * mm, table_top - 1.5 * mm, "||F(x)||₂")
        yy = table_top - 10 * mm
        for i, (name, item) in enumerate(rr.items()):
            c.setFillColor(colors.black if i % 2 == 0 else muted)
            c.setFont("DV", 7.8)
            c.drawString(22 * mm, yy, name)
            c.drawRightString(w - 55 * mm, yy, str(item["iterations"]))
            c.drawRightString(w - 22 * mm, yy, f"{item['residual']:.2e}")
            yy -= 6 * mm

        # Vector log plot in the lower half of the page.
        left, right, bottom, top = 31 * mm, w - 18 * mm, 43 * mm, 143 * mm
        values = [max(v, 1e-30) for item in rr.values() for v in item["history"]]
        log_lo, log_hi = math.floor(math.log10(min(values))), math.ceil(math.log10(max(values)))
        if log_lo == log_hi:
            log_lo -= 1
            log_hi += 1
        max_k = max(len(item["history"]) - 1 for item in rr.values())

        def px(k):
            return left + (right - left) * k / max(1, max_k)

        def py(value):
            return bottom + (top - bottom) * (math.log10(max(value, 1e-30)) - log_lo) / (log_hi - log_lo)

        c.setStrokeColor(colors.HexColor("#d8e0e7"))
        c.setLineWidth(0.45)
        for exp in range(log_lo, log_hi + 1):
            gy = bottom + (top - bottom) * (exp - log_lo) / (log_hi - log_lo)
            c.line(left, gy, right, gy)
            c.setFillColor(muted)
            c.setFont("DV", 7)
            c.drawRightString(left - 2 * mm, gy - 2, f"10^{exp}")
        xticks = sorted(set([0, max_k // 2, max_k]))
        for k in xticks:
            gx = px(k)
            c.line(gx, bottom, gx, top)
            c.setFillColor(muted)
            c.setFont("DV", 7)
            c.drawCentredString(gx, bottom - 4 * mm, str(k))
        c.setStrokeColor(navy)
        c.rect(left, bottom, right - left, top - bottom, stroke=1, fill=0)
        c.setFillColor(navy)
        c.setFont("DV", 8)
        c.drawCentredString((left + right) / 2, bottom - 9 * mm, "Номер итерации k")
        c.saveState()
        c.translate(14 * mm, (bottom + top) / 2)
        c.rotate(90)
        c.drawCentredString(0, 0, "Норма невязки ||F(xₖ)||₂, логарифмическая шкала")
        c.restoreState()
        for i, (name, item) in enumerate(rr.items()):
            color = colors.HexColor(PALETTE[i % len(PALETTE)])
            hist = item["history"]
            c.setStrokeColor(color)
            c.setFillColor(color)
            c.setLineWidth(1.35)
            path_obj = c.beginPath()
            path_obj.moveTo(px(0), py(hist[0]))
            for k, val in enumerate(hist[1:], 1):
                path_obj.lineTo(px(k), py(val))
            c.drawPath(path_obj, stroke=1, fill=0)
            for k, val in enumerate(hist):
                c.circle(px(k), py(val), 1.25, stroke=0, fill=1)
            lx, ly = 25 * mm + (i % 2) * 83 * mm, 30 * mm - (i // 2) * 5 * mm
            c.setStrokeColor(color)
            c.setLineWidth(1.8)
            c.line(lx, ly, lx + 6 * mm, ly)
            c.setFillColor(colors.HexColor("#243746"))
            c.setFont("DV", 6.5)
            c.drawString(lx + 8 * mm, ly - 2, name)
        c.showPage()
    c.save()


def main():
    outdir = Path("nonlinear_lab_results")
    outdir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for p in PROBLEMS:
        results = run_problem(p)
        all_results[p.key] = results
        svg_chart(p, results, outdir / f"{p.key}_convergence.svg")
    (outdir / "results.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    export_csv(outdir, all_results)
    create_pdf(outdir / "nonlinear_equations_lab_report.pdf", all_results)

    for p in PROBLEMS:
        print(f"\n{p.title}: {p.source}")
        for method, item in all_results[p.key].items():
            root = item["root"]
            print(f"  {method:28s} root={root}  k={item['iterations']:4d}  residual={item['residual']:.3e}  {item['status']}")
    print(f"\nОтчёт и графики: {outdir.resolve()}")


if __name__ == "__main__":
    main()
