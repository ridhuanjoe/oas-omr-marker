from __future__ import annotations
import argparse
import os
import csv
import cv2
import fitz
from omr.config import OMRConfig
from omr.pdf_utils import open_pdf, render_page, page_count
from omr.align import ORBAligner
from omr.layout import build_layout, save_layout, load_layout
from omr.mark import read_answers, read_student_number, annotate_page

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_image_jpg(path: str, bgr):
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])

def images_to_pdf(image_paths, out_pdf):
    doc = fitz.open()
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        page = doc.new_page(width=w, height=h)
        page.insert_image(page.rect, filename=p)
    doc.save(out_pdf)
    doc.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True, help="Blank OAS template PDF (2 pages recommended).")
    ap.add_argument("--key", required=True, help="Teacher key OAS PDF (2 pages).")
    ap.add_argument("--batch", required=True, help="Batch students PDF (2 pages per student).")
    ap.add_argument("--out", required=True, help="Output CSV path (student_number, score).")
    ap.add_argument("--annotated", required=True, help="Output annotated PDF path.")
    ap.add_argument("--layout", default="assets/layout.json", help="Layout cache file.")
    ap.add_argument("--rebuild-layout", action="store_true", help="Force rebuild layout from template.")
    args = ap.parse_args()

    cfg = OMRConfig()

    # --- Load template images ---
    tdoc = open_pdf(args.template)
    if len(tdoc) < 2:
        raise SystemExit("Template PDF should have 2 pages (page 1 + page 2).")

    t_p1 = render_page(tdoc, 0, cfg.dpi)
    t_p2 = render_page(tdoc, 1, cfg.dpi)

    align_p1 = ORBAligner(t_p1)
    align_p2 = ORBAligner(t_p2)

    # --- Layout ---
    if args.rebuild_layout or (not os.path.exists(args.layout)):
        layout = build_layout(t_p1, t_p2, cfg)
        ensure_dir(os.path.dirname(args.layout))
        save_layout(layout, args.layout)
    else:
        layout = load_layout(args.layout)

    # Questions
    qnums = list(range(1, 41)) + list(range(cfg.q2_start, cfg.q2_start + cfg.q2_count))

    # --- Read teacher key ---
    kdoc = open_pdf(args.key)
    if len(kdoc) < 2:
        raise SystemExit("Key PDF should have 2 pages.")
    k1 = render_page(kdoc, 0, cfg.dpi)
    k2 = render_page(kdoc, 1, cfg.dpi)

    k1a = align_p1.align(k1)
    k2a = align_p2.align(k2)

    key1, _ = read_answers(k1a, layout, list(range(1, 41)), cfg.answer_min_fill, cfg.multi_delta)
    key2, _ = read_answers(k2a, layout, list(range(cfg.q2_start, cfg.q2_start + cfg.q2_count)), cfg.answer_min_fill, cfg.multi_delta)

    key = {**key1, **key2}
    bad = [q for q, v in key.items() if v is None]
    if bad:
        raise SystemExit(f"Teacher key has blanks/ambiguous answers for questions: {bad}. Shade clearly and rescan.")

    # --- Batch marking ---
    bcount = page_count(args.batch)
    if bcount % 2 != 0:
        raise SystemExit(f"Batch PDF pages must be even (2 pages per student). Got {bcount} pages.")

    bdoc = open_pdf(args.batch)

    out_dir = os.path.dirname(args.out) or "."
    ann_dir = os.path.join(os.path.dirname(args.annotated) or ".", "_ann_imgs")
    ensure_dir(out_dir)
    ensure_dir(ann_dir)

    rows = []
    annotated_paths = []

    student_index = 0
    for i in range(0, bcount, 2):
        student_index += 1

        p1 = render_page(bdoc, i, cfg.dpi)
        p2 = render_page(bdoc, i+1, cfg.dpi)

        try:
            p1a = align_p1.align(p1)
        except Exception as e:
            raise SystemExit(f"Alignment failed on batch page {i+1}: {e}")
        try:
            p2a = align_p2.align(p2)
        except Exception as e:
            raise SystemExit(f"Alignment failed on batch page {i+2}: {e}")

        student_number = read_student_number(p1a, layout, cfg.digit_min_fill, cfg.multi_delta)

        ans1, _ = read_answers(p1a, layout, list(range(1, 41)), cfg.answer_min_fill, cfg.multi_delta)
        ans2, _ = read_answers(p2a, layout, list(range(cfg.q2_start, cfg.q2_start + cfg.q2_count)), cfg.answer_min_fill, cfg.multi_delta)
        answers = {**ans1, **ans2}

        score = sum(1 for q in qnums if answers.get(q) == key.get(q))

        rows.append({"student_number": student_number, "score": score})

        # Annotate
        ann1 = annotate_page(p1a, layout, answers, key, list(range(1, 41)))
        ann2 = annotate_page(p2a, layout, answers, key, list(range(cfg.q2_start, cfg.q2_start + cfg.q2_count)))

        p1_img = os.path.join(ann_dir, f"{student_index:03d}_p1.jpg")
        p2_img = os.path.join(ann_dir, f"{student_index:03d}_p2.jpg")
        save_image_jpg(p1_img, ann1)
        save_image_jpg(p2_img, ann2)
        annotated_paths.extend([p1_img, p2_img])

    # --- Write CSV (2 columns only) ---
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["student_number", "score"])
        for r in rows:
            w.writerow([r["student_number"], r["score"]])

    # --- Build annotated PDF ---
    images_to_pdf(annotated_paths, args.annotated)

    print(f"Done.\nCSV: {args.out}\nAnnotated PDF: {args.annotated}\nRows: {len(rows)}")

if __name__ == "__main__":
    main()
