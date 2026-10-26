#!/usr/bin/env python3
"""Numerical differentiation laboratory work.

The derivatives are approximated at x0 for five functions using the five
finite-difference formulas from the assignment.  For every function the
script saves one self-contained log-log SVG plot with the absolute errors of
all five methods.  The script uses only Python's standard library.

The photographed assignment does not show the evaluation point; x0=1 is
used by default.  It can be changed with --x0.
"""

from __future__ import annotations

import argparse
import html
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ScalarFunction = Callable[[float], float]
DifferenceMethod = Callable[[ScalarFunction, float, float], float]


@dataclass(frozen=True)
class FunctionSpec:
    name: str
    filename: str
    function: ScalarFunction
    derivative: ScalarFunction


def forward_difference(f: ScalarFunction, x: float, h: float) -> float:
    """f'(x) ≈ (f(x+h) - f(x)) / h."""
    return (f(x + h) - f(x)) / h


def backward_difference(f: ScalarFunction, x: float, h: float) -> float:
    """f'(x) ≈ (f(x) - f(x-h)) / h."""
    return (f(x) - f(x - h)) / h


def central_difference(f: ScalarFunction, x: float, h: float) -> float:
    """f'(x) ≈ (f(x+h) - f(x-h)) / (2h)."""
    return (f(x + h) - f(x - h)) / (2.0 * h)


def fourth_order_difference(f: ScalarFunction, x: float, h: float) -> float:
    """Fourth-order symmetric formula (formula 4 in the assignment)."""
    first = (f(x + h) - f(x - h)) / (2.0 * h)
    second = (f(x + 2.0 * h) - f(x - 2.0 * h)) / (4.0 * h)
    return (4.0 / 3.0) * first - (1.0 / 3.0) * second


def sixth_order_difference(f: ScalarFunction, x: float, h: float) -> float:
    """Sixth-order symmetric formula (formula 5 in the assignment)."""
    first = (f(x + h) - f(x - h)) / (2.0 * h)
    second = (f(x + 2.0 * h) - f(x - 2.0 * h)) / (4.0 * h)
    third = (f(x + 3.0 * h) - f(x - 3.0 * h)) / (6.0 * h)
    return (3.0 / 2.0) * first - (3.0 / 5.0) * second + (1.0 / 10.0) * third


METHODS: tuple[tuple[str, DifferenceMethod], ...] = (
    ("1. Прямая разность", forward_difference),
    ("2. Обратная разность", backward_difference),
    ("3. Центральная разность", central_difference),
    ("4. Симметричная 4-го порядка", fourth_order_difference),
    ("5. Симметричная 6-го порядка", sixth_order_difference),
)


FUNCTIONS: tuple[FunctionSpec, ...] = (
    FunctionSpec(
        name="f(x) = sin(x²)",
        filename="sin_x2",
        function=lambda x: math.sin(x**2),
        derivative=lambda x: 2.0 * x * math.cos(x**2),
    ),
    FunctionSpec(
        name="f(x) = cos(sin x)",
        filename="cos_sin_x",
        function=lambda x: math.cos(math.sin(x)),
        derivative=lambda x: -math.sin(math.sin(x)) * math.cos(x),
    ),
    FunctionSpec(
        name="f(x) = exp(sin(cos x))",
        filename="exp_sin_cos_x",
        function=lambda x: math.exp(math.sin(math.cos(x))),
        derivative=lambda x: -math.exp(math.sin(math.cos(x))) * math.cos(math.cos(x)) * math.sin(x),
    ),
    FunctionSpec(
        name="f(x) = ln(x + 3)",
        filename="ln_x_plus_3",
        function=lambda x: math.log(x + 3.0),
        derivative=lambda x: 1.0 / (x + 3.0),
    ),
    FunctionSpec(
        name="f(x) = sqrt(x + 3)",
        filename="sqrt_x_plus_3",
        function=lambda x: math.sqrt(x + 3.0),
        derivative=lambda x: 1.0 / (2.0 * math.sqrt(x + 3.0)),
    ),
)


def compute_errors(spec: FunctionSpec, x0: float, steps: list[float]) -> dict[str, list[float]]:
    exact = spec.derivative(x0)
    errors: dict[str, list[float]] = {}
    for label, method in METHODS:
        errors[label] = [abs(method(spec.function, x0, h) - exact) for h in steps]
    return errors


def log10(value: float) -> float:
    return math.log10(max(value, 1.0e-18))


def svg_text(x: float, y: float, text: str, *, size: int = 16, anchor: str = "start", weight: str = "400") -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="DejaVu Sans, sans-serif" '
        f'font-size="{size}px" font-weight="{weight}" text-anchor="{anchor}" fill="#202124">'
        f'{html.escape(text)}</text>'
    )


def save_plot(
    spec: FunctionSpec,
    x0: float,
    steps: list[float],
    errors: dict[str, list[float]],
    output_dir: Path,
) -> Path:
    """Write a self-contained SVG log-log plot without third-party packages."""
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = 1250, 760
    left, top, right, bottom = 112, 76, 925, 665
    plot_width, plot_height = right - left, bottom - top
    colors = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e")
    markers = ("circle", "square", "triangle", "diamond", "plus")

    x_min, x_max = min(log10(h) for h in steps), max(log10(h) for h in steps)
    positive_errors = [max(value, 1.0e-18) for values in errors.values() for value in values]
    y_min, y_max = min(log10(value) for value in positive_errors), max(log10(value) for value in positive_errors)
    x_min, x_max = math.floor(x_min) - 0.2, math.ceil(x_max) + 0.2
    y_min, y_max = math.floor(y_min) - 0.5, math.ceil(y_max) + 0.5

    def sx(value: float) -> float:
        return left + (log10(value) - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        return bottom - (log10(value) - y_min) / (y_max - y_min) * plot_height

    def line_path(values: list[float]) -> str:
        return " ".join(
            ("M" if i == 0 else "L") + f" {sx(steps[i]):.2f},{sy(values[i]):.2f}"
            for i in range(len(steps))
        )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 34, f"Абсолютная погрешность численного дифференцирования — {spec.name}", size=21, anchor="middle", weight="500"),
        svg_text(width / 2, 58, f"x₀ = {x0:g}; hₙ = 2 / 2ⁿ, n = 1,…,21", size=14, anchor="middle"),
    ]

    # Logarithmic grid and axes.
    x_tick_start, x_tick_end = math.ceil(x_min), math.floor(x_max)
    for exponent in range(x_tick_start, x_tick_end + 1):
        x = left + (exponent - x_min) / (x_max - x_min) * plot_width
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" stroke="#d9d9d9" stroke-dasharray="3,4"/>')
        label = "1" if exponent == 0 else f"10^{exponent}"
        parts.append(svg_text(x, bottom + 28, label, size=13, anchor="middle"))

    y_tick_start, y_tick_end = math.ceil(y_min), math.floor(y_max)
    for exponent in range(y_tick_start, y_tick_end + 1):
        y = bottom - (exponent - y_min) / (y_max - y_min) * plot_height
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#d9d9d9" stroke-dasharray="3,4"/>')
        label = "1" if exponent == 0 else f"10^{exponent}"
        parts.append(svg_text(left - 12, y + 5, label, size=13, anchor="end"))

    parts.extend([
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#202124" stroke-width="1.4"/>',
        svg_text((left + right) / 2, 720, "Шаг численного дифференцирования h", size=16, anchor="middle"),
        f'<g transform="translate(25 {(top + bottom) / 2:.2f}) rotate(-90)">{svg_text(0, 0, "Абсолютная погрешность ε(h)", size=16, anchor="middle")}</g>',
    ])

    for (label, _), color, marker in zip(METHODS, colors, markers):
        values = [max(value, 1.0e-18) for value in errors[label]]
        parts.append(f'<path d="{line_path(values)}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        for h, value in zip(steps, values):
            x, y = sx(h), sy(value)
            if marker == "circle":
                parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
            elif marker == "square":
                parts.append(f'<rect x="{x-4:.2f}" y="{y-4:.2f}" width="8" height="8" fill="{color}"/>')
            elif marker == "triangle":
                parts.append(f'<path d="M{x:.2f},{y-5:.2f} L{x+5:.2f},{y+4:.2f} L{x-5:.2f},{y+4:.2f} Z" fill="{color}"/>')
            elif marker == "diamond":
                parts.append(f'<path d="M{x:.2f},{y-5:.2f} L{x+5:.2f},{y:.2f} L{x:.2f},{y+5:.2f} L{x-5:.2f},{y:.2f} Z" fill="{color}"/>')
            else:
                parts.append(f'<path d="M{x-5:.2f},{y:.2f} H{x+5:.2f} M{x:.2f},{y-5:.2f} V{y+5:.2f}" stroke="{color}" stroke-width="2.2"/>')

    legend_x, legend_y = 958, 135
    parts.append(svg_text(legend_x, legend_y - 30, "Методы", size=16, weight="500"))
    for index, ((label, _), color) in enumerate(zip(METHODS, colors)):
        y = legend_y + index * 42
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+28}" y2="{y}" stroke="{color}" stroke-width="2.2"/>')
        parts.append(svg_text(legend_x + 40, y + 5, label, size=13))

    parts.append('</svg>')
    path = output_dir / f"{spec.filename}.svg"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def write_summary(
    output_dir: Path,
    x0: float,
    steps: list[float],
    all_errors: dict[str, dict[str, list[float]]],
) -> Path:
    lines = [
        "# Численное дифференцирование",
        "",
        f"Точка вычисления: `x₀ = {x0:g}`.",
        "Шаги: `hₙ = 2 / 2ⁿ`, `n = 1, ..., 21`.",
        "",
        "На каждом графике показаны абсолютные погрешности пяти формул из задания в логарифмическом масштабе по обеим осям.",
        "",
        "## Аналитические производные",
        "",
        "- `sin(x²)`: `2x cos(x²)`",
        "- `cos(sin(x))`: `-sin(sin(x)) cos(x)`",
        "- `exp(sin(cos(x)))`: `-exp(sin(cos(x))) cos(cos(x)) sin(x)`",
        "- `ln(x+3)`: `1/(x+3)`",
        "- `sqrt(x+3)`: `1/(2 sqrt(x+3))`",
        "",
        "## Графики",
        "",
    ]
    for spec in FUNCTIONS:
        lines.append(f"- `{spec.filename}.svg` — {spec.name}")

    lines.extend(["", "## Минимальная достигнутая погрешность", ""])
    for spec in FUNCTIONS:
        lines.append(f"### {spec.name}")
        for label, _ in METHODS:
            values = all_errors[spec.filename][label]
            i = min(range(len(values)), key=values.__getitem__)
            lines.append(f"- {label}: `{values[i]:.3e}` при `h = {steps[i]:.8g}`")
        lines.append("")

    path = output_dir / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--x0",
        type=float,
        default=1.0,
        help="точка, в которой вычисляется производная (по умолчанию: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("numerical_differentiation_results"),
        help="каталог для SVG-графиков и summary.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = [2.0 / (2.0**n) for n in range(1, 22)]

    all_errors: dict[str, dict[str, list[float]]] = {}
    for spec in FUNCTIONS:
        errors = compute_errors(spec, args.x0, steps)
        all_errors[spec.filename] = errors
        save_plot(spec, args.x0, steps, errors, args.output_dir)

    summary_path = write_summary(args.output_dir, args.x0, steps, all_errors)
    print(f"Сохранено 5 графиков и сводка: {summary_path}")


if __name__ == "__main__":
    main()
