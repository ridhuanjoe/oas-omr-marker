from __future__ import annotations
import fitz  # PyMuPDF
import numpy as np
import cv2

def open_pdf(path: str) -> fitz.Document:
    return fitz.open(path)

def page_count(path: str) -> int:
    with fitz.open(path) as d:
        return len(d)

def render_page(doc: fitz.Document, page_index: int, dpi: int) -> np.ndarray:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    page = doc[page_index]
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    # Convert to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img
