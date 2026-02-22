from __future__ import annotations
import numpy as np
import cv2
from typing import Dict, Tuple, Optional

def _fill_ratio(gray: np.ndarray, cx: float, cy: float, r: float) -> float:
    h, w = gray.shape[:2]
    cx_i, cy_i = int(round(cx)), int(round(cy))
    r_in = max(3, int(round(r * 0.55)))  # inner circle to avoid outline
    x1, x2 = max(0, cx_i - r_in), min(w, cx_i + r_in)
    y1, y2 = max(0, cy_i - r_in), min(h, cy_i + r_in)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    _, bw = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    mh, mw = roi.shape[:2]
    mask = np.zeros((mh, mw), dtype=np.uint8)
    cv2.circle(mask, (mw // 2, mh // 2), int(r_in * 0.85), 255, -1)

    filled = (bw[mask == 255] == 255).sum()
    total = (mask == 255).sum()
    return float(filled) / float(total) if total else 0.0

def _pick(scores: Dict[str, float], min_fill: float, multi_delta: float) -> Tuple[Optional[str], str]:
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_opt, best = items[0]
    second = items[1][1] if len(items) > 1 else 0.0

    if best < min_fill:
        return None, "blank"
    if (best - second) < multi_delta:
        return None, "multi"
    return best_opt, "ok"

def read_answers(aligned_bgr: np.ndarray, layout: dict, qnums: list, min_fill: float, multi_delta: float):
    gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    out = {}
    status = {}
    for q in qnums:
        bubbles = layout["questions"][str(q)]
        scores = {b["opt"]: _fill_ratio(gray, b["x"], b["y"], b["r"]) for b in bubbles}
        choice, st = _pick(scores, min_fill=min_fill, multi_delta=multi_delta)
        out[q] = choice
        status[q] = st
    return out, status

def read_student_number(aligned_bgr: np.ndarray, layout: dict, min_fill: float, multi_delta: float) -> str:
    gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    digits = []
    for col in layout["student_number"]["columns"]:
        scores = {str(d["digit"]): _fill_ratio(gray, d["x"], d["y"], d["r"]) for d in col}
        choice, _st = _pick(scores, min_fill=min_fill, multi_delta=multi_delta)
        digits.append(choice if choice is not None else "")
    sn = "".join(digits).strip()
    return sn if sn else "UNKNOWN"

def annotate_page(aligned_bgr: np.ndarray, layout: dict, answers: dict, key: dict, qnums: list):
    img = aligned_bgr.copy()
    for q in qnums:
        chosen = answers.get(q, None)
        bubbles = layout["questions"][str(q)]

        for b in bubbles:
            cv2.circle(img, (int(b["x"]), int(b["y"])), int(b["r"]), (180, 180, 180), 1)

        if chosen is None:
            continue

        bsel = next((b for b in bubbles if b["opt"] == chosen), None)
        if not bsel:
            continue
        correct = (chosen == key.get(q))
        color = (0, 200, 0) if correct else (0, 0, 255)
        cv2.circle(img, (int(bsel["x"]), int(bsel["y"])), int(bsel["r"]*1.2), color, 3)
    return img
