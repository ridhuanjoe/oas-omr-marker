from __future__ import annotations
import json
import numpy as np
import cv2
from .config import OMRConfig

def _rel_to_abs(region, w, h):
    x1, y1, x2, y2 = region
    return int(x1*w), int(y1*h), int(x2*w), int(y2*h)

def detect_circles(bgr: np.ndarray, cfg: OMRConfig):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=cfg.hough_dp,
        minDist=cfg.hough_min_dist,
        param1=cfg.hough_param1,
        param2=cfg.hough_param2,
        minRadius=cfg.hough_min_radius,
        maxRadius=cfg.hough_max_radius,
    )
    if circles is None:
        return []
    return circles[0].tolist()  # [x,y,r] float

def _cluster_by_gap(vals, gap):
    vals = sorted(vals)
    clusters = []
    cur = [vals[0]]
    for v in vals[1:]:
        if v - cur[-1] <= gap:
            cur.append(v)
        else:
            clusters.append(cur)
            cur = [v]
    clusters.append(cur)
    centers = [sum(c)/len(c) for c in clusters]
    return centers, clusters

def build_layout(template_p1: np.ndarray, template_p2: np.ndarray, cfg: OMRConfig) -> dict:
    """
    Builds a layout (bubble map) from template pages using circle detection + clustering.
    Saves bubble circles for:
      - student number grid
      - Q1–40 (page 1)
      - Q41–45 (page 2)
    """
    h, w = template_p1.shape[:2]

    # --- PAGE 1: Answers region (Q1–40) ---
    ax1, ay1, ax2, ay2 = _rel_to_abs(cfg.p1_answers_region, w, h)
    p1_ans = template_p1[ay1:ay2, ax1:ax2]
    circles = detect_circles(p1_ans, cfg)

    if len(circles) < 150:
        raise RuntimeError("Layout build failed: too few circles detected in page-1 answers region.")

    circles_abs = [(c[0]+ax1, c[1]+ay1, c[2]) for c in circles]
    ys = [c[1] for c in circles_abs]

    # Get ~10 row centers
    gap = 8
    centers_y = None
    for _ in range(12):
        cy, _ = _cluster_by_gap(ys, gap)
        if len(cy) <= cfg.q1_rows:
            centers_y = sorted(cy)[:cfg.q1_rows]
            break
        gap += 3
    if centers_y is None or len(centers_y) < cfg.q1_rows:
        centers_y = sorted(np.quantile(ys, np.linspace(0.05, 0.95, cfg.q1_rows)).tolist())

    layout_q = {}
    dy = 18  # row tolerance
    for row_i, y0 in enumerate(centers_y):
        row_circles = [c for c in circles_abs if abs(c[1] - y0) <= dy]
        row_circles.sort(key=lambda c: c[0])
        if not row_circles:
            continue

        # Split into blocks by large gaps
        groups = []
        cur = [row_circles[0]]
        for c in row_circles[1:]:
            if c[0] - cur[-1][0] > 70:
                groups.append(cur)
                cur = [c]
            else:
                cur.append(c)
        if cur:
            groups.append(cur)

        groups = sorted(groups, key=len, reverse=True)[:cfg.q1_blocks]
        groups = sorted(groups, key=lambda g: sum(x[0] for x in g)/len(g))

        for block_i, g in enumerate(groups):
            g = sorted(g, key=lambda c: c[0])[:len(cfg.options)]
            if len(g) != len(cfg.options):
                continue
            qnum = block_i * cfg.q1_rows + (row_i + 1)
            layout_q[str(qnum)] = [
                {"opt": cfg.options[j], "x": float(g[j][0]), "y": float(g[j][1]), "r": float(g[j][2])}
                for j in range(len(cfg.options))
            ]

    # --- PAGE 1: Student number region ---
    sx1, sy1, sx2, sy2 = _rel_to_abs(cfg.p1_student_number_region, w, h)
    p1_sn = template_p1[sy1:sy2, sx1:sx2]
    sn_circles = detect_circles(p1_sn, cfg)
    if len(sn_circles) < 60:
        raise RuntimeError("Layout build failed: too few circles detected in student-number region.")
    sn_abs = [(c[0]+sx1, c[1]+sy1, c[2]) for c in sn_circles]
    sn_abs.sort(key=lambda c: c[0])

    xs = [c[0] for c in sn_abs]
    gap = 12
    col_centers = None
    clusters = None
    for _ in range(12):
        cx, cl = _cluster_by_gap(xs, gap)
        if 6 <= len(cx) <= 12:
            col_centers, clusters = cx, cl
            break
        gap += 3
    if col_centers is None:
        col_centers, clusters = _cluster_by_gap(xs, 25)

    columns = []
    for x0 in sorted(col_centers):
        col = [c for c in sn_abs if abs(c[0] - x0) <= 14]
        col.sort(key=lambda c: c[1])
        if len(col) < 8:
            continue
        ys_col = [c[1] for c in col]
        y_centers, _ = _cluster_by_gap(ys_col, 10)
        y_centers = sorted(y_centers)[:10]

        digit_circles = []
        for yc in y_centers:
            nearest = min(col, key=lambda c: abs(c[1] - yc))
            digit_circles.append(nearest)

        digit_circles = sorted(digit_circles, key=lambda c: c[1])[:10]
        columns.append([
            {"digit": int(cfg.digit_order[i]), "x": float(digit_circles[i][0]), "y": float(digit_circles[i][1]), "r": float(digit_circles[i][2])}
            for i in range(min(10, len(digit_circles)))
        ])

    # --- PAGE 2: Q41–45 (top of first column) ---
    h2, w2 = template_p2.shape[:2]
    bx1, by1, bx2, by2 = _rel_to_abs(cfg.p2_first_column_region, w2, h2)
    p2_block = template_p2[by1:by2, bx1:bx2]
    c2 = detect_circles(p2_block, cfg)
    if len(c2) < 40:
        raise RuntimeError("Layout build failed: too few circles detected in page-2 first-column region.")
    c2_abs = [(c[0]+bx1, c[1]+by1, c[2]) for c in c2]

    ys2 = sorted([c[1] for c in c2_abs])
    y_centers2, _ = _cluster_by_gap(ys2, 9)
    y_centers2 = sorted(y_centers2)[:cfg.q2_count]

    layout_q2 = {}
    for i, y0 in enumerate(y_centers2):
        row = [c for c in c2_abs if abs(c[1] - y0) <= 14]
        row.sort(key=lambda c: c[0])
        row = row[:len(cfg.options)]
        if len(row) != len(cfg.options):
            continue
        qnum = cfg.q2_start + i
        layout_q2[str(qnum)] = [
            {"opt": cfg.options[j], "x": float(row[j][0]), "y": float(row[j][1]), "r": float(row[j][2])}
            for j in range(len(cfg.options))
        ]

    layout = {
        "meta": {"dpi": cfg.dpi},
        "student_number": {"columns": columns},
        "questions": {**layout_q, **layout_q2},
    }

    if "1" not in layout["questions"] or str(cfg.q2_start + cfg.q2_count - 1) not in layout["questions"]:
        raise RuntimeError("Layout build failed: missing required questions (check regions in config.py).")

    return layout

def save_layout(layout: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2)

def load_layout(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
