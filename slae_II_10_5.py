#!/usr/bin/env python3
"""Solve exercise II.10.5 and compare direct and iterative methods.

Dependencies: numpy and reportlab.  A user-defined system can be supplied as
JSON: {"A": [[...], ...], "b": [...], "x0": [...] (optional)}.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


METHOD_COLORS = ["#2364aa", "#d1495b", "#2a9d8f", "#8a5fbf", "#ed8b00", "#52616b", "#bc5090"]
ITERATIVE_KEYS = ("jacobi", "gauss_seidel", "sor", "gradient", "minimal_residual", "cg", "bicgstab")
METHOD_LABELS = {
    "gauss": "Гаусс с выбором главного элемента",
    "lu": "LU-разложение с перестановками",
    "jacobi": "Якоби",
    "gauss_seidel": "Зейдель",
    "sor": "Верхняя релаксация",
    "gradient": "Градиентный спуск",
    "minimal_residual": "Минимальные невязки",
    "cg": "Сопряжённые градиенты",
    "bicgstab": "BiCGSTAB",
}


def selected_variant_m() -> tuple[np.ndarray, np.ndarray]:
    """Variant м): symmetric pentadiagonal matrix, n=20, rhs f_i=i."""
    n = 20
    A = np.zeros((n, n), dtype=float)
    np.fill_diagonal(A, 10.0)
    for i in range(n):
        if i + 1 < n:
            A[i, i + 1] = A[i + 1, i] = 1.0
        if i + 2 < n:
            A[i, i + 2] = A[i + 2, i] = 0.1
    b = np.arange(1.0, n + 1.0)
    return A, b


def residual_norm(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    return float(np.linalg.norm(b - A @ x, ord=2))


def stopping_threshold(b: np.ndarray, rtol: float, atol: float) -> float:
    return atol + rtol * float(np.linalg.norm(b, ord=2))


def gaussian_elimination(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Gaussian elimination with partial pivoting."""
    M = np.array(A, dtype=float, copy=True)
    rhs = np.array(b, dtype=float, copy=True)
    n = len(rhs)
    for k in range(n):
        pivot = k + int(np.argmax(np.abs(M[k:, k])))
        if abs(M[pivot, k]) <= np.finfo(float).eps * max(1.0, np.linalg.norm(M, ord=np.inf)):
            raise np.linalg.LinAlgError("Матрица вырождена или численно близка к вырожденной")
        if pivot != k:
            M[[k, pivot]] = M[[pivot, k]]
            rhs[[k, pivot]] = rhs[[pivot, k]]
        for i in range(k + 1, n):
            factor = M[i, k] / M[k, k]
            M[i, k] = 0.0
            M[i, k + 1 :] -= factor * M[k, k + 1 :]
            rhs[i] -= factor * rhs[k]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (rhs[i] - M[i, i + 1 :] @ x[i + 1 :]) / M[i, i]
    return x


def lu_factor_partial_pivot(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return P,L,U satisfying P@A=L@U, using partial pivoting."""
    U = np.array(A, dtype=float, copy=True)
    n = U.shape[0]
    L = np.eye(n)
    P = np.eye(n)
    for k in range(n):
        pivot = k + int(np.argmax(np.abs(U[k:, k])))
        if abs(U[pivot, k]) <= np.finfo(float).eps * max(1.0, np.linalg.norm(U, ord=np.inf)):
            raise np.linalg.LinAlgError("Матрица вырождена или численно близка к вырожденной")
        if pivot != k:
            U[[k, pivot]] = U[[pivot, k]]
            P[[k, pivot]] = P[[pivot, k]]
            if k:
                L[[k, pivot], :k] = L[[pivot, k], :k]
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
            U[i, k] = 0.0
    return P, L, U


def lu_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    P, L, U = lu_factor_partial_pivot(A)
    rhs = P @ b
    n = len(rhs)
    y = np.zeros(n)
    for i in range(n):
        y[i] = rhs[i] - L[i, :i] @ y[:i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1 :] @ x[i + 1 :]) / U[i, i]
    return x


def jacobi_iteration(A, b, x0, tol, max_iter):
    D = np.diag(A)
    if np.any(np.abs(D) < np.finfo(float).eps):
        raise ValueError("Метод Якоби неприменим: на диагонали есть ноль")
    R = A - np.diagflat(D)
    x = x0.copy()
    hist = [residual_norm(A, b, x)]
    for k in range(1, max_iter + 1):
        x = (b - R @ x) / D
        hist.append(residual_norm(A, b, x))
        if hist[-1] <= tol:
            return x, hist, "сошёлся"
    return x, hist, "достигнут лимит итераций"


def gauss_seidel_iteration(A, b, x0, tol, max_iter, omega=1.0):
    n = len(b)
    if np.any(np.abs(np.diag(A)) < np.finfo(float).eps):
        raise ValueError("Нулевой диагональный элемент")
    x = x0.copy()
    hist = [residual_norm(A, b, x)]
    for _ in range(max_iter):
        old = x.copy()
        for i in range(n):
            raw = (b[i] - A[i, :i] @ x[:i] - A[i, i + 1 :] @ old[i + 1 :]) / A[i, i]
            x[i] = (1.0 - omega) * old[i] + omega * raw
        hist.append(residual_norm(A, b, x))
        if not np.all(np.isfinite(x)):
            return x, hist, "расходимость: получены нечисловые значения"
        if hist[-1] <= tol:
            return x, hist, "сошёлся"
    return x, hist, "достигнут лимит итераций"


def estimate_sor_omega(A: np.ndarray) -> tuple[float, float]:
    D = np.diag(A)
    if np.any(np.abs(D) < np.finfo(float).eps):
        return 1.0, math.nan
    B = np.eye(len(D)) - A / D[:, None]
    rho = float(np.max(np.abs(np.linalg.eigvals(B))))
    if rho < 1.0:
        return 2.0 / (1.0 + math.sqrt(max(0.0, 1.0 - rho * rho))), rho
    return 1.0, rho


def steepest_descent(A, b, x0, tol, max_iter):
    """Steepest descent for the SPD quadratic 1/2*x.T*A*x-b.T*x."""
    x = x0.copy()
    r = b - A @ x
    hist = [float(np.linalg.norm(r))]
    for _ in range(max_iter):
        Ar = A @ r
        denom = float(r @ Ar)
        if denom <= np.finfo(float).eps * max(1.0, float(r @ r)):
            return x, hist, "остановлен: нарушено условие положительной определённости"
        alpha = float(r @ r) / denom
        x += alpha * r
        r = b - A @ x
        hist.append(float(np.linalg.norm(r)))
        if hist[-1] <= tol:
            return x, hist, "сошёлся"
    return x, hist, "достигнут лимит итераций"


def minimal_residual(A, b, x0, tol, max_iter):
    """Minimal residual step x+=alpha*r, alpha=(Ar,r)/(Ar,Ar)."""
    x = x0.copy()
    r = b - A @ x
    hist = [float(np.linalg.norm(r))]
    for _ in range(max_iter):
        Ar = A @ r
        denom = float(Ar @ Ar)
        if denom <= np.finfo(float).tiny:
            return x, hist, "остановлен: норма A·r близка к нулю"
        alpha = float(Ar @ r) / denom
        x += alpha * r
        r = b - A @ x
        hist.append(float(np.linalg.norm(r)))
        if hist[-1] <= tol:
            return x, hist, "сошёлся"
    return x, hist, "достигнут лимит итераций"


def conjugate_gradient(A, b, x0, tol, max_iter):
    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    rr = float(r @ r)
    hist = [math.sqrt(rr)]
    for _ in range(max_iter):
        Ap = A @ p
        denom = float(p @ Ap)
        if denom <= np.finfo(float).eps * max(1.0, rr):
            return x, hist, "остановлен: матрица не выглядит положительно определённой"
        alpha = rr / denom
        x += alpha * p
        r -= alpha * Ap
        rr_new = float(r @ r)
        hist.append(math.sqrt(rr_new))
        if hist[-1] <= tol:
            return x, hist, "сошёлся"
        beta = rr_new / rr
        p = r + beta * p
        rr = rr_new
    return x, hist, "достигнут лимит итераций"


def bicgstab(A, b, x0, tol, max_iter):
    x = x0.copy()
    r = b - A @ x
    r_hat = r.copy()
    rho_old = alpha = omega = 1.0
    v = np.zeros_like(r)
    p = np.zeros_like(r)
    hist = [float(np.linalg.norm(r))]
    tiny = np.finfo(float).tiny
    for _ in range(max_iter):
        rho = float(r_hat @ r)
        if abs(rho) <= tiny:
            return x, hist, "остановлен: breakdown (rho≈0)"
        beta = (rho / rho_old) * (alpha / omega)
        p = r + beta * (p - omega * v)
        v = A @ p
        den = float(r_hat @ v)
        if abs(den) <= tiny:
            return x, hist, "остановлен: breakdown (знаменатель alpha≈0)"
        alpha = rho / den
        s = r - alpha * v
        if float(np.linalg.norm(s)) <= tol:
            x += alpha * p
            hist.append(residual_norm(A, b, x))
            return x, hist, "сошёлся"
        t = A @ s
        tt = float(t @ t)
        if tt <= tiny:
            return x, hist, "остановлен: breakdown (t·t≈0)"
        omega = float(t @ s) / tt
        x += alpha * p + omega * s
        r = s - omega * t
        hist.append(float(np.linalg.norm(r)))
        if hist[-1] <= tol:
            return x, hist, "сошёлся"
        if abs(omega) <= tiny:
            return x, hist, "остановлен: breakdown (omega≈0)"
        rho_old = rho
    return x, hist, "достигнут лимит итераций"


def load_system(path: Path | None):
    if path is None:
        A, b = selected_variant_m()
        return A, b, np.zeros(len(b)), "Вариант м) из практикума II.10.5"
    data = json.loads(path.read_text(encoding="utf-8"))
    A = np.asarray(data["A"], dtype=float)
    b = np.asarray(data["b"], dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or b.shape != (A.shape[0],):
        raise ValueError("JSON должен содержать квадратную A и вектор b согласованной размерности")
    x0 = np.asarray(data.get("x0", np.zeros(len(b))), dtype=float)
    if x0.shape != b.shape:
        raise ValueError("Вектор x0 должен иметь ту же размерность, что и b")
    return A, b, x0, data.get("name", path.stem)


def solve_all(A: np.ndarray, b: np.ndarray, x0: np.ndarray, rtol: float, atol: float, max_iter: int):
    tol = stopping_threshold(b, rtol, atol)
    results = {}
    direct_solvers: tuple[tuple[str, Callable], ...] = (("gauss", gaussian_elimination), ("lu", lu_solve))
    for key, solver in direct_solvers:
        try:
            x = solver(A, b)
            results[key] = {"x": x, "history": [residual_norm(A, b, x)], "status": "прямой метод", "iterations": 1}
        except (np.linalg.LinAlgError, ValueError) as exc:
            results[key] = {"x": np.full_like(b, np.nan), "history": [], "status": str(exc), "iterations": 0}

    try:
        symmetry_error = float(np.linalg.norm(A - A.T, ord=np.inf))
        symmetric = symmetry_error <= 1e-12 * max(1.0, float(np.linalg.norm(A, ord=np.inf)))
        eigmin = float(np.linalg.eigvalsh(A)[0]) if symmetric else math.nan
        positive_definite = symmetric and eigmin > 0.0
    except np.linalg.LinAlgError:
        symmetric, positive_definite, eigmin = False, False, math.nan

    iterative: list[tuple[str, Callable]] = [
        ("jacobi", jacobi_iteration),
        ("gauss_seidel", gauss_seidel_iteration),
    ]
    omega, rho_j = estimate_sor_omega(A)
    try:
        x, hist, status = gauss_seidel_iteration(A, b, x0.copy(), tol, max_iter, omega=omega)
        results["sor"] = {"x": x, "history": hist, "status": status, "iterations": len(hist) - 1, "omega": omega, "rho_jacobi": rho_j}
    except (ValueError, np.linalg.LinAlgError) as exc:
        results["sor"] = {"x": x0.copy(), "history": [], "status": str(exc), "iterations": 0, "omega": omega, "rho_jacobi": rho_j}

    if positive_definite:
        iterative.extend([("gradient", steepest_descent), ("cg", conjugate_gradient)])
    else:
        for key in ("gradient", "cg"):
            results[key] = {"x": np.full_like(b, np.nan), "history": [], "status": "неприменим к этой системе: требуется симметричная положительно определённая матрица", "iterations": 0}
    iterative.extend([("minimal_residual", minimal_residual), ("bicgstab", bicgstab)])

    for key, method in iterative:
        if key == "sor":
            continue
        try:
            x, hist, status = method(A, b, x0.copy(), tol, max_iter)
            results[key] = {"x": x, "history": hist, "status": status, "iterations": len(hist) - 1}
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            results[key] = {"x": np.full_like(b, np.nan), "history": [], "status": str(exc), "iterations": 0}
    return results, tol, symmetric, positive_definite, eigmin


def write_svg(path: Path, results: dict, threshold: float):
    series = [(key, results[key]["history"]) for key in ITERATIVE_KEYS if results[key]["history"]]
    width, height = 1120, 680
    left, right, top, bottom = 90, 800, 55, 575
    xs_max = max(len(h) - 1 for _, h in series)
    ys = [max(v, 1e-30) for _, hist in series for v in hist]
    ymin, ymax = math.floor(math.log10(min(ys))), math.ceil(math.log10(max(ys)))
    if ymin == ymax:
        ymin -= 1
        ymax += 1

    def sx(i):
        return left + (right - left) * i / max(1, xs_max)

    def sy(v):
        return bottom - (bottom - top) * (math.log10(max(v, 1e-30)) - ymin) / (ymax - ymin)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    parts.append('<text x="560" y="28" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="20" fill="#1d3145">Сходимость итерационных методов</text>')
    for ex in range(ymin, ymax + 1):
        y = bottom - (bottom - top) * (ex - ymin) / (ymax - ymin)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#d8e0e7"/>')
        parts.append(f'<text x="{left-10}" y="{y+5:.2f}" text-anchor="end" font-family="DejaVu Sans,sans-serif" font-size="13" fill="#34495e">10^{ex}</text>')
    x_ticks = sorted(set([0, 1, max(2, xs_max // 4), max(3, xs_max // 2), max(4, 3 * xs_max // 4), xs_max]))
    for i in x_ticks:
        if i > xs_max:
            continue
        x = sx(i)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" stroke="#e9edf0"/>')
        parts.append(f'<text x="{x:.2f}" y="{bottom+22}" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="12" fill="#34495e">{i}</text>')
    parts.append(f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="none" stroke="#34495e"/>')
    parts.append(f'<text x="{(left+right)/2}" y="{bottom+50}" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="15" fill="#243746">Номер итерации k</text>')
    parts.append(f'<text x="22" y="{(top+bottom)/2}" transform="rotate(-90 22 {(top+bottom)/2})" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="15" fill="#243746">Норма невязки ||b − Axₖ||₂</text>')
    if 10**ymin <= threshold <= 10**ymax:
        y = sy(threshold)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#333333" stroke-dasharray="6,5"/>')
        parts.append(f'<text x="{right-4}" y="{y-6:.2f}" text-anchor="end" font-family="DejaVu Sans,sans-serif" font-size="12" fill="#333333">порог останова</text>')
    for idx, (key, hist) in enumerate(series):
        col = METHOD_COLORS[idx % len(METHOD_COLORS)]
        d = " ".join(("M" if i == 0 else "L") + f"{sx(i):.2f},{sy(v):.2f}" for i, v in enumerate(hist))
        parts.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2.3"/>')
        for i, value in enumerate(hist):
            parts.append(f'<circle cx="{sx(i):.2f}" cy="{sy(value):.2f}" r="2.6" fill="{col}"/>')
        ly = 88 + idx * 38
        parts.append(f'<line x1="850" y1="{ly}" x2="880" y2="{ly}" stroke="{col}" stroke-width="3"/>')
        parts.append(f'<text x="890" y="{ly+5}" font-family="DejaVu Sans,sans-serif" font-size="14" fill="#243746">{METHOD_LABELS[key]}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def export_csv(path: Path, results: dict):
    keys = [key for key in ITERATIVE_KEYS if results[key]["history"]]
    max_len = max(len(results[k]["history"]) for k in keys)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["iteration", *[METHOD_LABELS[k] for k in keys]])
        for i in range(max_len):
            writer.writerow([i, *[results[k]["history"][i] if i < len(results[k]["history"]) else "" for k in keys]])


def create_pdf(path: Path, results: dict, A: np.ndarray, b: np.ndarray, source_name: str, threshold: float, sor_omega: float):
    font_regular = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font_bold = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    pdfmetrics.registerFont(TTFont("DV", str(font_regular)))
    pdfmetrics.registerFont(TTFont("DVB", str(font_bold)))
    c = canvas.Canvas(str(path), pagesize=A4)
    page_w, page_h = A4
    navy = colors.HexColor("#19324a")
    muted = colors.HexColor("#566879")
    c.setTitle("Решение СЛАУ II.10.5 — вариант м)")
    c.setAuthor("Дернович Арсений; Русецкий Егор")

    # Page 1: problem statement (two lines) and method comparison.
    c.setFillColor(navy)
    c.setFont("DVB", 19)
    c.drawCentredString(page_w / 2, page_h - 22 * mm, "Практическая работа II.10.5")
    c.setFont("DVB", 15)
    c.drawCentredString(page_w / 2, page_h - 31 * mm, "Решение системы линейных уравнений")
    c.setFillColor(muted)
    c.setFont("DV", 9)
    c.drawCentredString(page_w / 2, page_h - 38 * mm, f"Выбран вариант м) • {source_name} • размер {len(b)}×{len(b)}")
    c.setFont("DV", 9)
    c.drawCentredString(page_w / 2, page_h - 45 * mm, "Авторы: Дернович Арсений, Русецкий Егор")

    y = page_h - 59 * mm
    c.setFillColor(navy)
    c.setFont("DVB", 11)
    c.drawString(20 * mm, y, "Постановка задачи")
    y -= 7 * mm
    c.setFillColor(colors.black)
    c.setFont("DV", 9)
    c.drawString(20 * mm, y, "A x = f;  aᵢᵢ = 10,  aᵢ,ᵢ₊₁ = aᵢ₊₁,ᵢ = 1,  aᵢ,ᵢ₊₂ = aᵢ₊₂,ᵢ = 0,1,  fᵢ = i.")
    y -= 6 * mm
    c.drawString(20 * mm, y, "n = 20; остальные элементы A равны нулю. Начальное приближение итерационных методов: x⁽⁰⁾ = 0.")
    y -= 11 * mm
    c.setFillColor(navy)
    c.setFont("DVB", 11)
    c.drawString(20 * mm, y, "Сравнение методов")
    y -= 7 * mm
    c.setFillColor(colors.HexColor("#eaf0f4"))
    c.rect(20 * mm, y - 4 * mm, page_w - 40 * mm, 7 * mm, fill=1, stroke=0)
    c.setFillColor(navy)
    c.setFont("DVB", 8)
    c.drawString(22 * mm, y - 1.5 * mm, "Метод")
    c.drawRightString(page_w - 75 * mm, y - 1.5 * mm, "Итерации")
    c.drawRightString(page_w - 22 * mm, y - 1.5 * mm, "||b−Ax||₂")
    y -= 9 * mm
    order = ("gauss", "lu", "jacobi", "gauss_seidel", "sor", "gradient", "minimal_residual", "cg", "bicgstab")
    for i, key in enumerate(order):
        res = results[key]
        c.setFillColor(colors.black if i % 2 == 0 else muted)
        c.setFont("DV", 7.8)
        c.drawString(22 * mm, y, METHOD_LABELS[key])
        c.drawRightString(page_w - 75 * mm, y, str(res["iterations"]) if res["iterations"] else "—")
        final_r = residual_norm(A, b, res["x"]) if np.all(np.isfinite(res["x"])) else math.nan
        c.drawRightString(page_w - 22 * mm, y, f"{final_r:.2e}" if math.isfinite(final_r) else "не применим")
        y -= 5.2 * mm
    c.showPage()

    # Page 2: vector convergence chart.
    c.setFillColor(navy)
    c.setFont("DVB", 16)
    c.drawCentredString(page_w / 2, page_h - 19 * mm, "Убывание невязки итерационных методов")
    x_left, x_right = 34 * mm, page_w - 18 * mm
    y_bottom, y_top = 68 * mm, page_h - 38 * mm
    histories = [(key, results[key]["history"]) for key in ITERATIVE_KEYS if results[key]["history"]]
    all_res = [max(value, 1e-30) for _, hist in histories for value in hist]
    log_min, log_max = math.floor(math.log10(min(all_res))), math.ceil(math.log10(max(all_res)))
    if log_min == log_max:
        log_min -= 1
        log_max += 1
    max_k = max(len(hist) - 1 for _, hist in histories)

    def pdf_x(k):
        return x_left + (x_right - x_left) * k / max(1, max_k)

    def pdf_y(val):
        return y_bottom + (y_top - y_bottom) * (math.log10(max(val, 1e-30)) - log_min) / (log_max - log_min)

    c.setStrokeColor(colors.HexColor("#d8e0e7"))
    c.setLineWidth(0.5)
    for exponent in range(log_min, log_max + 1):
        yy = y_bottom + (y_top - y_bottom) * (exponent - log_min) / (log_max - log_min)
        c.line(x_left, yy, x_right, yy)
        c.setFont("DV", 8)
        c.setFillColor(muted)
        c.drawRightString(x_left - 3 * mm, yy - 2, f"10^{exponent}")
    tick_vals = sorted(set([0, 1, max_k // 4, max_k // 2, (3 * max_k) // 4, max_k]))
    for k in tick_vals:
        xx = pdf_x(k)
        c.setStrokeColor(colors.HexColor("#edf0f2"))
        c.line(xx, y_bottom, xx, y_top)
        c.setFillColor(muted)
        c.drawCentredString(xx, y_bottom - 5 * mm, str(k))
    c.setStrokeColor(navy)
    c.rect(x_left, y_bottom, x_right - x_left, y_top - y_bottom, stroke=1, fill=0)
    c.setFillColor(navy)
    c.setFont("DV", 9)
    c.drawCentredString((x_left + x_right) / 2, y_bottom - 11 * mm, "Номер итерации k")
    c.saveState()
    c.translate(13 * mm, (y_bottom + y_top) / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, "Норма невязки ||b − Axₖ||₂ (логарифмическая шкала)")
    c.restoreState()
    if 10**log_min <= threshold <= 10**log_max:
        yy = pdf_y(threshold)
        c.setStrokeColor(colors.HexColor("#333333"))
        c.setDash(4, 3)
        c.line(x_left, yy, x_right, yy)
        c.setDash()
        c.setFillColor(colors.HexColor("#333333"))
        c.setFont("DV", 7.5)
        c.drawRightString(x_right - 2 * mm, yy + 2, "порог останова")
    for idx, (key, hist) in enumerate(histories):
        col = colors.HexColor(METHOD_COLORS[idx % len(METHOD_COLORS)])
        c.setStrokeColor(col)
        c.setFillColor(col)
        c.setLineWidth(1.5)
        p = c.beginPath()
        p.moveTo(pdf_x(0), pdf_y(hist[0]))
        for k, value in enumerate(hist[1:], start=1):
            p.lineTo(pdf_x(k), pdf_y(value))
        c.drawPath(p, stroke=1, fill=0)
        for k, value in enumerate(hist):
            c.circle(pdf_x(k), pdf_y(value), 1.7, stroke=0, fill=1)
    legend_x, legend_y = 23 * mm, 42 * mm
    for idx, (key, hist) in enumerate(histories):
        row, col_idx = divmod(idx, 4)
        xx = legend_x + col_idx * 46 * mm
        yy = legend_y - row * 8 * mm
        col = colors.HexColor(METHOD_COLORS[idx % len(METHOD_COLORS)])
        c.setStrokeColor(col)
        c.setLineWidth(2)
        c.line(xx, yy, xx + 6 * mm, yy)
        c.setFillColor(colors.HexColor("#243746"))
        c.setFont("DV", 6.8)
        c.drawString(xx + 8 * mm, yy - 2.5, METHOD_LABELS[key])
    c.setFillColor(muted)
    c.setFont("DV", 8)
    c.drawString(20 * mm, 18 * mm, "Сходимость принята при ||b−Ax⁽ᵏ⁾||₂ ≤ atol + rtol·||b||₂; значения atol=0, rtol=1·10⁻¹⁰.")
    c.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", type=Path, help="JSON-файл с ключами A, b и необязательным x0/name")
    parser.add_argument("--output-dir", type=Path, default=Path("slae_II_10_5_results"))
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--max-iter", type=int, default=20000)
    args = parser.parse_args()
    if args.rtol < 0 or args.atol < 0 or args.max_iter <= 0:
        parser.error("rtol и atol должны быть неотрицательными, max-iter — положительным")

    A, b, x0, source_name = load_system(args.system)
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        parser.error("A и b должны состоять из конечных чисел")
    if np.linalg.matrix_rank(A) < len(b):
        parser.error("Матрица системы вырождена")
    results, tol, symmetric, positive_definite, eigmin = solve_all(A, b, x0, args.rtol, args.atol, args.max_iter)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.set_printoptions(precision=12, suppress=True)
    omega = float(results["sor"].get("omega", 1.0))
    (args.output_dir / "system_used.json").write_text(json.dumps({"name": source_name, "A": A.tolist(), "b": b.tolist(), "x0": x0.tolist()}, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "results.json").write_text(json.dumps({key: {"solution": res["x"].tolist(), "iterations": res["iterations"], "residual_history": res["history"], "status": res["status"]} for key, res in results.items()}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_svg(args.output_dir / "residual_convergence.svg", results, tol)
    export_csv(args.output_dir / "residual_history.csv", results)
    create_pdf(args.output_dir / "II_10_5_report.pdf", results, A, b, source_name, tol, omega)

    print(f"Система: {source_name}; n={len(b)}; symmetric={symmetric}; SPD={positive_definite}; lambda_min={eigmin:.12g}")
    print(f"Критерий останова: ||b-Ax||₂ <= {tol:.4e}; SOR omega={omega:.10g}")
    print(f"{'Метод':34s} {'k':>8s} {'||b-Ax||₂':>15s}  Статус")
    for key in ("gauss", "lu", *ITERATIVE_KEYS):
        res = results[key]
        rn = residual_norm(A, b, res["x"]) if np.all(np.isfinite(res["x"])) else math.nan
        print(f"{METHOD_LABELS[key]:34s} {res['iterations']:8d} {rn:15.6e}  {res['status']}")
    print("\nРешение (Гаусс):")
    for i, value in enumerate(results["gauss"]["x"], 1):
        print(f"x{i:02d} = {value:.12f}")
    print(f"\n2-норма обусловленности: {np.linalg.cond(A, 2):.12g}")
    print(f"Результаты сохранены в: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
