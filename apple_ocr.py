"""Extract text from PDFs with Apple's Vision OCR framework."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PageText:
    number: int
    text: str


def _require_macos() -> None:
    if sys.platform != "darwin":
        raise RuntimeError("Apple OCR requires macOS.")


def _frameworks():
    _require_macos()
    try:
        import Quartz
        import Vision
        from Foundation import NSURL
    except ImportError as exc:
        raise RuntimeError(
            "Apple Vision frameworks are not installed. From this project run:\n"
            "  pip install -r requirements.txt"
        ) from exc
    return Quartz, Vision, NSURL


def _perform(handler, request) -> None:
    result = handler.performRequests_error_([request], None)
    if isinstance(result, tuple):
        ok, error = result
        if not ok:
            message = error.localizedDescription() if error is not None else "unknown error"
            raise RuntimeError(f"Vision OCR failed: {message}")
        return
    if result is False:
        raise RuntimeError("Vision OCR failed.")


def _page_box(quartz, page):
    crop = quartz.CGPDFPageGetBoxRect(page, quartz.kCGPDFCropBox)
    if crop.size.width > 1 and crop.size.height > 1:
        return crop, quartz.kCGPDFCropBox
    return quartz.CGPDFPageGetBoxRect(page, quartz.kCGPDFMediaBox), quartz.kCGPDFMediaBox


def _render_page(quartz, document, page_number: int, scale: float):
    page = quartz.CGPDFDocumentGetPage(document, page_number)
    if page is None:
        raise RuntimeError(f"PDF has no page {page_number}.")

    box, box_type = _page_box(quartz, page)
    width = box.size.width
    height = box.size.height
    rotation = int(quartz.CGPDFPageGetRotationAngle(page)) % 360
    if rotation in (90, 270):
        width, height = height, width

    pixel_width = max(1, int(round(width * scale)))
    pixel_height = max(1, int(round(height * scale)))
    color_space = quartz.CGColorSpaceCreateDeviceRGB()
    context = quartz.CGBitmapContextCreate(
        None,
        pixel_width,
        pixel_height,
        8,
        0,
        color_space,
        quartz.kCGImageAlphaPremultipliedLast | quartz.kCGBitmapByteOrder32Big,
    )
    if context is None:
        raise RuntimeError(f"Could not render page {page_number}.")

    dest = quartz.CGRectMake(0, 0, pixel_width, pixel_height)
    quartz.CGContextSetRGBFillColor(context, 1, 1, 1, 1)
    quartz.CGContextFillRect(context, dest)
    # This transform only scales a page down to fit, then centers it.
    # Scale the context first so the page is rasterized at `scale` pixels per point.
    quartz.CGContextScaleCTM(context, scale, scale)
    point_dest = quartz.CGRectMake(0, 0, width, height)
    transform = quartz.CGPDFPageGetDrawingTransform(page, box_type, point_dest, 0, True)
    quartz.CGContextConcatCTM(context, transform)
    quartz.CGContextDrawPDFPage(context, page)
    image = quartz.CGBitmapContextCreateImage(context)
    if image is None:
        raise RuntimeError(f"Could not create an image for page {page_number}.")
    return image


def _lines_in_reading_order(observations, min_confidence: float) -> list[str]:
    items = []
    for observation in observations:
        candidates = observation.topCandidates_(1)
        if not candidates:
            continue
        candidate = candidates[0]
        confidence = float(candidate.confidence())
        if confidence < min_confidence:
            continue
        text = str(candidate.string() or "").strip()
        if not text:
            continue
        box = observation.boundingBox()
        items.append(
            {
                "text": text,
                "x": float(box.origin.x),
                "top": float(box.origin.y + box.size.height),
                "height": float(box.size.height),
            }
        )

    if not items:
        return []

    items.sort(key=lambda item: (-item["top"], item["x"]))
    rows: list[list[dict]] = [[items[0]]]
    for item in items[1:]:
        reference = rows[-1][0]
        same_row = abs(item["top"] - reference["top"]) <= max(0.012, reference["height"] * 0.6)
        if same_row:
            rows[-1].append(item)
        else:
            rows.append([item])

    lines = []
    for row in rows:
        row.sort(key=lambda item: item["x"])
        lines.append(" ".join(item["text"] for item in row))
    return lines


def _recognize(vision, image, languages: list[str], fast: bool, min_confidence: float) -> str:
    request = vision.VNRecognizeTextRequest.alloc().init()
    level = (
        vision.VNRequestTextRecognitionLevelFast
        if fast
        else vision.VNRequestTextRecognitionLevelAccurate
    )
    request.setRecognitionLevel_(level)
    request.setUsesLanguageCorrection_(True)
    if languages:
        request.setRecognitionLanguages_(languages)
        if request.respondsToSelector_("setAutomaticallyDetectsLanguage:"):
            request.setAutomaticallyDetectsLanguage_(False)
    elif request.respondsToSelector_("setAutomaticallyDetectsLanguage:"):
        request.setAutomaticallyDetectsLanguage_(True)

    handler = vision.VNImageRequestHandler.alloc().initWithCGImage_options_(image, None)
    _perform(handler, request)
    return "\n".join(_lines_in_reading_order(request.results() or [], min_confidence))


def open_pdf(path: Path):
    quartz, _vision, NSURL = _frameworks()
    url = NSURL.fileURLWithPath_(str(path.resolve()))
    document = quartz.CGPDFDocumentCreateWithURL(url)
    if document is None:
        raise RuntimeError(f"Could not open PDF: {path}")
    if not quartz.CGPDFDocumentIsUnlocked(document):
        raise RuntimeError(f"PDF is encrypted and cannot be read: {path}")
    count = int(quartz.CGPDFDocumentGetNumberOfPages(document))
    if count < 1:
        raise RuntimeError(f"PDF has no pages: {path}")
    return document, count


def extract_pdf(
    path: Path,
    *,
    scale: float = 2.0,
    languages: list[str] | None = None,
    fast: bool = False,
    min_confidence: float = 0.0,
) -> list[PageText]:
    """Render each page and return the text Apple Vision recognized."""
    if scale <= 0:
        raise ValueError("scale must be greater than 0.")
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1.")

    quartz, vision, _NSURL = _frameworks()
    document, count = open_pdf(path)
    language_list = list(languages) if languages else ["en-US"]
    pages: list[PageText] = []
    for number in range(1, count + 1):
        image = _render_page(quartz, document, number, scale)
        text = _recognize(vision, image, language_list, fast, min_confidence)
        pages.append(PageText(number=number, text=text))
    return pages


def pages_to_text(pages: list[PageText]) -> str:
    blocks = []
    for page in pages:
        body = page.text.strip()
        blocks.append(f"[Page {page.number}]\n{body}" if body else f"[Page {page.number}]")
    return "\n\n".join(blocks) + "\n"
