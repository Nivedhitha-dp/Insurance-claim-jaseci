"""Turning files into positioned text lines.

PDF  -> PyMuPDF text layer (exact; falls back to OCR when a PDF has no text layer)
image-> RapidOCR (PaddleOCR models on ONNX, CPU-only, open source), optional enhancement pass
text -> as-is

OCR is the expensive step (~10-40s/page on CPU), so results are cached on disk by
content hash + mode. Re-processing a claim or re-uploading the same file is free.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Optional

from .config import settings
from .models import TextLine

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
TEXT_EXT = {".txt", ".eml", ".md"}

_ocr_engine = None
_ocr_lock = threading.Lock()


def file_kind(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext in IMAGE_EXT:
        return "image"
    if ext in TEXT_EXT:
        return "text"
    return "unsupported"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- PDF
def pdf_lines(path: Path) -> list[TextLine]:
    import pymupdf

    out: list[TextLine] = []
    with pymupdf.open(path) as doc:
        for pno, page in enumerate(doc):
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    text = "".join(s["text"] for s in line["spans"]).replace("�", "-").strip()
                    if text:
                        x0, y0, x1, y1 = line["bbox"]
                        out.append(TextLine(text=text, conf=1.0, x0=x0, y0=y0, x1=x1, y1=y1, page=pno))
    return out


def pdf_to_image(path: Path) -> "object":
    import numpy as np
    import pymupdf

    with pymupdf.open(path) as doc:
        pix = doc[0].get_pixmap(dpi=150)
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()


# ----------------------------------------------------------------------------- OCR
def _engine():
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR

        _ocr_engine = RapidOCR()
    return _ocr_engine


def enhance(img, skew_deg: float = 0.0):
    """Scanner clean-up for a second OCR pass: deskew -> denoise -> contrast -> upscale."""
    import cv2

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if abs(skew_deg) > 0.2:
        h, w = gray.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), skew_deg, 1.0)
        gray = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.resize(gray, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _skew(boxes) -> float:
    """Page skew (degrees) = median angle of the top edge of wide text boxes."""
    import math
    import statistics

    angles = [math.degrees(math.atan2(b[1][1] - b[0][1], b[1][0] - b[0][0]))
              for b in boxes if (b[1][0] - b[0][0]) > 80]
    angles = [a for a in angles if abs(a) < 8]
    return round(statistics.median(angles), 2) if angles else 0.0


def ocr_page(path: Path, enhanced: bool = False, digest: Optional[str] = None) -> dict:
    """OCR one page -> {lines, skew_deg}. Box coordinates are geometrically de-skewed, so
    'value to the right of label' still works on a page that went through the scanner crooked."""
    import math

    digest = digest or sha256(path)
    cache = settings.cache_dir / f"ocr2_{digest[:24]}_{'enh' if enhanced else 'std'}.json"
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        return {"lines": [TextLine(**l) for l in data["lines"]], "skew_deg": data["skew_deg"]}

    import cv2
    import numpy as np

    img = pdf_to_image(path) if path.suffix.lower() == ".pdf" else cv2.imdecode(
        np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    scale = 1.0
    if enhanced:
        img, scale = enhance(img, ocr_page(path, False, digest)["skew_deg"]), 1.6
    with _ocr_lock:  # one shared onnxruntime session
        result, _ = _engine()(img, use_cls=False)
    result = result or []
    skew = 0.0 if enhanced else _skew([r[0] for r in result])
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    rad = math.radians(-skew)
    cos, sin = math.cos(rad), math.sin(rad)
    lines = []
    for box, text, conf in result:
        pts = [((x - cx) * cos - (y - cy) * sin + cx, (x - cx) * sin + (y - cy) * cos + cy) for x, y in box]
        xs, ys = [p[0] / scale for p in pts], [p[1] / scale for p in pts]
        lines.append(TextLine(text=text.strip(), conf=round(float(conf), 3),
                              x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys)))
    lines.sort(key=lambda l: ((l.y0 + l.y1) / 2 // 8, l.x0))
    cache.write_text(json.dumps({"lines": [l.model_dump() for l in lines], "skew_deg": skew}), encoding="utf-8")
    return {"lines": lines, "skew_deg": skew}


def ocr_lines(path: Path, enhanced: bool = False, digest: Optional[str] = None) -> list[TextLine]:
    return ocr_page(path, enhanced, digest)["lines"]


def lines_to_text(lines: list[TextLine]) -> str:
    """Reading-order text: group lines into rows by vertical centre, left to right."""
    rows: list[list[TextLine]] = []
    for l in sorted(lines, key=lambda l: (l.page, (l.y0 + l.y1) / 2)):
        c = (l.y0 + l.y1) / 2
        if rows and rows[-1][0].page == l.page and abs((rows[-1][0].y0 + rows[-1][0].y1) / 2 - c) < 6:
            rows[-1].append(l)
        else:
            rows.append([l])
    return "\n".join("   ".join(x.text for x in sorted(r, key=lambda l: l.x0)) for r in rows)


def mean_conf(lines: list[TextLine]) -> Optional[float]:
    # weight by length so stray specks don't dominate
    total = sum(len(l.text) for l in lines)
    return round(sum(l.conf * len(l.text) for l in lines) / total, 3) if total else None
