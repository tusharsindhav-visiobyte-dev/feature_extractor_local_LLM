#!/usr/bin/env python3
"""Extract travel features from PDF page images with a local vision model.

The model sees the rendered pages. This script does not run OCR and does not
send PDF bytes to Ollama. The saved JSON is a TravelDocument, the same contract
used by the existing feature files and by data_collection.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from extract_features import (
    CLASSIFICATION_KEYS,
    EMAIL_RE,
    PHONE_RE,
    SENTINELS,
    SYSTEM_PROMPT,
    _merge,
    parse_json_object,
)
from travel_schema import DocumentType, TravelDocument, to_travel_document


logger = logging.getLogger("extract_features_vlm")

DEFAULT_HOST = "http://127.0.0.1:11435"
DEFAULT_MODEL = "qwen2.5vl:7b"
DEFAULT_OUTPUT_DIR = Path("data/QWEN_VL")
DEFAULT_DPI = 150
DEFAULT_MAX_PAGES = 4
DEFAULT_MAX_IMAGE_BYTES = 6_000_000
BENCHMARK_COLUMNS = (
    "filename",
    "pages",
    "pdf_open_time",
    "rendering_time",
    "image_bytes",
    "llm_request_time",
    "parsing_time",
    "total_time",
    "input_tokens",
    "output_tokens",
    "prompt_eval_time",
    "generation_time",
    "status",
    "error",
)

VISION_SYSTEM_PROMPT = SYSTEM_PROMPT + """
You are looking at photographs of the original document pages, not at OCR text.
Use the visual layout, labels, logos, headings, and nearby words together.
Never decide the document type from one number on its own.
A booking reference, PNR, ticket number, phone number, or membership id is not a document type.
Distinguish a flight ticket from a train ticket, a train ticket from a bus ticket, and a hotel confirmation from a mixed itinerary.
Distinguish a booking reference from a flight or train number, and an airline PNR from a railway PNR.
Passengers and lead guests are travelers. Do not treat a support agent, a hotel property, a phone number, an email address, or a member id as a passenger.
If the document type cannot be determined from the pages, use unknown.
"""

_PLACE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "code": {"type": "string"},
        "terminal": {"type": "string"},
        "gate": {"type": "string"},
        "datetime": {"type": "string"},
    },
    "required": ["name", "code", "terminal", "gate", "datetime"],
}
_DOCUMENT_TYPE_ENUM = sorted({item.value for item in DocumentType} | {"invoice", "visa", "package"})
_MODE_ENUM = ["flight", "train", "bus", "ferry", "cruise", "car", "other"]
VISION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "document_type": {"type": "string", "enum": _DOCUMENT_TYPE_ENUM},
        "document_subtype": {"type": "string"},
        "document_title": {"type": "string"},
        "booking_references": {"type": "array", "items": {"type": "string"}},
        "confirmation_codes": {"type": "array", "items": {"type": "string"}},
        "ticket_numbers": {"type": "array", "items": {"type": "string"}},
        "booking_status": {"type": "string"},
        "booking_date": {"type": "string"},
        "issue_date": {"type": "string"},
        "travelers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                    "ticket_number": {"type": "string"},
                    "seat": {"type": "string"},
                    "coach": {"type": "string"},
                },
                "required": ["name", "role", "ticket_number", "seat", "coach"],
            },
        },
        "transport": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": _MODE_ENUM},
                    "carrier": {"type": "string"},
                    "service_number": {"type": "string"},
                    "from": _PLACE_SCHEMA,
                    "to": _PLACE_SCHEMA,
                    "seat": {"type": "string"},
                    "coach": {"type": "string"},
                    "cabin": {"type": "string"},
                    "platform": {"type": "string"},
                    "boarding_time": {"type": "string"},
                    "status": {"type": "string"},
                    "fare": {"type": "string"},
                },
                "required": [
                    "mode",
                    "carrier",
                    "service_number",
                    "from",
                    "to",
                    "seat",
                    "coach",
                    "cabin",
                    "platform",
                    "boarding_time",
                    "status",
                    "fare",
                ],
            },
        },
        "lodging": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "address": {"type": "string"},
                    "check_in": {"type": "string"},
                    "check_out": {"type": "string"},
                    "nights": {"type": "string"},
                    "room_type": {"type": "string"},
                    "guests": {"type": "string"},
                    "confirmation_code": {"type": "string"},
                    "total": {"type": "string"},
                    "cancellation": {"type": "string"},
                },
                "required": [
                    "name",
                    "address",
                    "check_in",
                    "check_out",
                    "nights",
                    "room_type",
                    "guests",
                    "confirmation_code",
                    "total",
                    "cancellation",
                ],
            },
        },
        "car_rentals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "vehicle": {"type": "string"},
                    "pickup_location": {"type": "string"},
                    "pickup_datetime": {"type": "string"},
                    "dropoff_location": {"type": "string"},
                    "dropoff_datetime": {"type": "string"},
                    "confirmation_code": {"type": "string"},
                    "total": {"type": "string"},
                },
                "required": [
                    "company",
                    "vehicle",
                    "pickup_location",
                    "pickup_datetime",
                    "dropoff_location",
                    "dropoff_datetime",
                    "confirmation_code",
                    "total",
                ],
            },
        },
        "emails": {"type": "array", "items": {"type": "string"}},
        "phones": {"type": "array", "items": {"type": "string"}},
        "baggage": {"type": "string"},
        "notes": {"type": "array", "items": {"type": "string"}},
        "currency": {"type": "string"},
        "price_total": {"type": "string"},
        "payment_status": {"type": "string"},
    },
    "required": [
        "document_type",
        "document_subtype",
        "document_title",
        "booking_references",
        "confirmation_codes",
        "ticket_numbers",
        "booking_status",
        "booking_date",
        "issue_date",
        "travelers",
        "transport",
        "lodging",
        "car_rentals",
        "emails",
        "phones",
        "baggage",
        "notes",
        "currency",
        "price_total",
        "payment_status",
    ],
}
_INTERMEDIATE_KEYS = (
    "transport",
    "lodging",
    "travelers",
    "confirmation_codes",
    "important_notes",
    "price",
    "price_total",
)
_UNIFIED_KEYS = ("flights", "hotels", "trains", "buses")


class InvalidModelResponse(RuntimeError):
    """The model answered, but the JSON could not be used as a travel document."""


class ExtractionFailure(RuntimeError):
    """A document failed after any timings already collected."""

    def __init__(self, message: str, outcome: Outcome):
        super().__init__(message)
        self.outcome = outcome


@dataclass
class RenderedPage:
    number: int
    jpeg: bytes


@dataclass
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    prompt_eval_time: float | None = None
    generation_time: float | None = None
    total_duration: float | None = None

    def add(self, other: Usage) -> None:
        self.input_tokens = _sum_optional(self.input_tokens, other.input_tokens)
        self.output_tokens = _sum_optional(self.output_tokens, other.output_tokens)
        self.prompt_eval_time = _sum_optional(self.prompt_eval_time, other.prompt_eval_time)
        self.generation_time = _sum_optional(self.generation_time, other.generation_time)
        self.total_duration = _sum_optional(self.total_duration, other.total_duration)


@dataclass
class Outcome:
    features: dict | None = None
    usage: Usage = field(default_factory=Usage)
    request_time: float = 0.0
    parse_time: float = 0.0


@dataclass
class BenchmarkRow:
    filename: str
    pages: int = 0
    pdf_open_time: float = 0.0
    rendering_time: float = 0.0
    image_bytes: int = 0
    llm_request_time: float = 0.0
    parsing_time: float = 0.0
    total_time: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    prompt_eval_time: float | None = None
    generation_time: float | None = None
    status: str = "failure"
    error: str = ""


def nanoseconds_to_seconds(value) -> float | None:
    if value is None:
        return None
    return float(value) / 1_000_000_000


def _sum_optional(current: float | int | None, incoming: float | int | None):
    if incoming is None:
        return current
    if current is None:
        return incoming
    return current + incoming


def _response_value(response, name: str):
    if isinstance(response, dict):
        return response.get(name)
    return getattr(response, name, None)


def usage_from_response(response) -> Usage:
    return Usage(
        input_tokens=_response_value(response, "prompt_eval_count"),
        output_tokens=_response_value(response, "eval_count"),
        prompt_eval_time=nanoseconds_to_seconds(_response_value(response, "prompt_eval_duration")),
        generation_time=nanoseconds_to_seconds(_response_value(response, "eval_duration")),
        total_duration=nanoseconds_to_seconds(_response_value(response, "total_duration")),
    )


def build_extraction_prompt(page_numbers: list[int], total_pages: int) -> str:
    listed = ", ".join(str(number) for number in page_numbers)
    if page_numbers == list(range(1, total_pages + 1)):
        scope = (
            f"The attached images are all {total_pages} page(s) of one document, in order. "
            "Read every page before deciding the document type or copying the final facts. "
            "Do not assume the first page contains everything."
        )
    else:
        scope = (
            f"The attached images are pages {listed} of a {total_pages}-page document, in that order. "
            "Extract only facts visible on these pages. Another step merges the pages. "
            "Do not invent facts from pages that are not attached, and do not drop a fact that is visible here."
        )
    return f"""Extract the useful travel and booking features visible in this document.
booking_references means a booking reference, PNR, or record locator.
service_number means only the flight, train, or bus number. Do not copy a PNR into service_number.
Create one transport item per flight, train, bus, ferry, or cruise. Do not put a hotel in transport.
Put each hotel in lodging. Put each car rental in car_rentals.
notes are short conditions copied from the document, such as a cancellation deadline.
document_type must be one of: {", ".join(_DOCUMENT_TYPE_ENUM)}.
Use unknown when the pages do not show the type reliably.
Use itinerary only when the pages show more than one kind of booking.
If the document is an invoice for a flight, hotel, train, bus, car, or cruise, set document_type to that booking and document_subtype to invoice.
Use an empty string or an empty list when a fact is not visible. Never invent a person, place, price, time, or reference.
Preserve dates, times, and identifiers as printed.

{scope}
"""


def _clean_sentinels(value, key: str | None = None):
    if isinstance(value, dict):
        return {name: _clean_sentinels(item, name) for name, item in value.items()}
    if isinstance(value, list):
        return [_clean_sentinels(item) for item in value]
    if isinstance(value, str) and key not in CLASSIFICATION_KEYS and value.strip().lower() in SENTINELS:
        return None
    return value


def _looks_like_contact(value: str) -> bool:
    text = " ".join(value.split())
    if EMAIL_RE.fullmatch(text):
        return True
    if re.search(r"[A-Za-z]", text):
        return False
    if not PHONE_RE.fullmatch(text):
        return False
    digits = re.sub(r"\D", "", text)
    return 10 <= len(digits) <= 15


def _drop_contact_names(payload: dict) -> None:
    for key in ("travelers", "passengers"):
        people = payload.get(key)
        if not isinstance(people, list):
            continue
        kept = []
        for person in people:
            name = person.get("name") if isinstance(person, dict) else None
            if isinstance(name, str) and _looks_like_contact(name):
                continue
            kept.append(person)
        payload[key] = kept


def _place_from_item(item: dict, prefix: str) -> dict:
    nested = item.get(prefix)
    if isinstance(nested, str):
        place = {"name": nested}
    elif isinstance(nested, dict):
        place = dict(nested)
    else:
        place = {}
    aliases = {
        "name": f"{prefix}_place",
        "code": f"{prefix}_code",
        "terminal": f"{prefix}_terminal",
        "gate": f"{prefix}_gate",
        "datetime": f"{prefix}_datetime",
    }
    for destination, source in aliases.items():
        if not place.get(destination) and item.get(source):
            place[destination] = item[source]
    return place


def _normalize_transport(item: dict) -> dict:
    normalized = dict(item)
    normalized["from"] = _place_from_item(normalized, "from")
    normalized["to"] = _place_from_item(normalized, "to")
    if not normalized.get("cabin_or_class") and normalized.get("cabin"):
        normalized["cabin_or_class"] = normalized["cabin"]
    return normalized


def prepare_payload(raw: dict) -> dict:
    """Move model fields onto the shape to_travel_document already accepts."""
    payload = _clean_sentinels(raw)
    if not isinstance(payload, dict):
        raise InvalidModelResponse("The model returned JSON that is not an object.")
    transport = payload.get("transport")
    if isinstance(transport, list):
        payload["transport"] = [_normalize_transport(item) if isinstance(item, dict) else item for item in transport]
    price = dict(payload.get("price") or {}) if isinstance(payload.get("price"), dict) else {}
    if not price.get("currency") and payload.get("currency"):
        price["currency"] = payload["currency"]
    if not price.get("total") and payload.get("price_total"):
        price["total"] = payload["price_total"]
    if not price.get("payment_status") and payload.get("payment_status"):
        price["payment_status"] = payload["payment_status"]
    if price:
        payload["price"] = price
    contacts = dict(payload.get("contacts") or {}) if isinstance(payload.get("contacts"), dict) else {}
    if not contacts.get("emails") and payload.get("emails"):
        contacts["emails"] = payload["emails"]
    if not contacts.get("phones") and payload.get("phones"):
        contacts["phones"] = payload["phones"]
    if contacts:
        payload["contacts"] = contacts
    _drop_contact_names(payload)
    _dedupe_people(payload)
    if any(key in payload for key in _INTERMEDIATE_KEYS):
        for key in _UNIFIED_KEYS:
            payload.pop(key, None)
    return payload


def _dedupe_people(payload: dict) -> None:
    for key in ("travelers", "passengers"):
        people = payload.get(key)
        if not isinstance(people, list):
            continue
        merged: list[dict] = []
        by_name: dict[str, dict] = {}
        for person in people:
            if not isinstance(person, dict):
                continue
            name = " ".join(str(person.get("name") or "").lower().split())
            if not name:
                merged.append(person)
                continue
            current = by_name.get(name)
            if current is None:
                by_name[name] = person
                merged.append(person)
                continue
            for field_name, value in person.items():
                if value and not current.get(field_name):
                    current[field_name] = value
        payload[key] = merged


def merge_observations(parts: list[dict]) -> dict:
    prepared = [prepare_payload(part) for part in parts if isinstance(part, dict)]
    if not prepared:
        raise InvalidModelResponse("The model did not return a JSON object.")
    merged = prepared[0]
    for part in prepared[1:]:
        merged = _merge(merged, part)
    _dedupe_people(merged)
    return merged


def parse_model_response(raw: str) -> dict:
    try:
        parsed = parse_json_object(raw)
    except RuntimeError as exc:
        raise InvalidModelResponse(str(exc)) from exc
    if not isinstance(parsed, dict):
        raise InvalidModelResponse("The model returned JSON that is not an object.")
    return parsed


def validate_features(payload: dict, source_filename: str | None = None) -> dict:
    prepared = prepare_payload(payload)
    declared = prepared.get("document_type")
    source_text = "boarding pass" if isinstance(declared, str) and declared.strip().lower() == "boarding_pass" else ""
    try:
        document = to_travel_document(prepared, source_text=source_text, source_filename=source_filename)
        checked = TravelDocument.model_validate(document.model_dump(mode="json"))
    except (RuntimeError, TypeError, ValueError) as exc:
        raise InvalidModelResponse(f"Travel document failed validation: {exc}") from exc
    return checked.model_dump(mode="json")


def render_pdf_pages(path: Path, *, dpi: float, jpeg_quality: int) -> tuple[list[RenderedPage], float, float]:
    """Open one PDF and return JPEG pages in order. No files are left behind."""
    if dpi <= 0:
        raise ValueError("dpi must be greater than 0.")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100.")
    import pymupdf

    opened = time.perf_counter()
    try:
        document = pymupdf.open(path)
    except Exception as exc:
        raise RuntimeError(f"Could not open PDF {path.name}: {exc}") from exc
    try:
        if getattr(document, "needs_pass", False) and not document.authenticate(""):
            raise RuntimeError(f"PDF is password-protected and cannot be read: {path.name}")
        if document.page_count < 1:
            raise RuntimeError(f"PDF has no pages: {path.name}")
        open_time = time.perf_counter() - opened
        started = time.perf_counter()
        matrix = pymupdf.Matrix(dpi / 72, dpi / 72)
        pages: list[RenderedPage] = []
        for number, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            pages.append(RenderedPage(number=number, jpeg=pixmap.tobytes("jpeg", jpg_quality=jpeg_quality)))
            del pixmap
        return pages, open_time, time.perf_counter() - started
    finally:
        document.close()


def batch_pages(pages: list[RenderedPage], max_pages: int, max_image_bytes: int) -> list[list[RenderedPage]]:
    """Group pages without dropping any. One oversized page is still sent."""
    if max_pages < 1:
        raise ValueError("max_pages must be at least 1.")
    if max_image_bytes < 1:
        raise ValueError("max_image_bytes must be at least 1.")
    if not pages:
        return []
    total_bytes = sum(len(page.jpeg) for page in pages)
    if len(pages) <= max_pages and total_bytes <= max_image_bytes:
        return [list(pages)]

    batches: list[list[RenderedPage]] = []
    current: list[RenderedPage] = []
    current_bytes = 0
    for page in pages:
        overflows = bool(current) and (
            len(current) >= max_pages or current_bytes + len(page.jpeg) > max_image_bytes
        )
        if overflows:
            batches.append(current)
            current = []
            current_bytes = 0
        current.append(page)
        current_bytes += len(page.jpeg)
    if current:
        batches.append(current)
    return batches


def _ollama_modules():
    try:
        import httpx
        import ollama
    except ImportError as exc:
        raise RuntimeError(
            "The Ollama client is not installed. From this project run:\n  pip install -r requirements.txt"
        ) from exc
    return ollama, httpx


def build_client(host: str, timeout: float):
    ollama, httpx = _ollama_modules()
    return ollama.Client(host=host, timeout=httpx.Timeout(timeout))


def _message_text(response) -> str:
    message = _response_value(response, "message")
    content = _response_value(message, "content") if message is not None else None
    if isinstance(content, str):
        return content
    return ""


def ask_vlm(client, model: str, prompt: str, images: list[bytes], *, num_ctx: int, num_predict: int) -> tuple[str, Usage]:
    ollama, httpx = _ollama_modules()
    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt, "images": images},
            ],
            format=VISION_RESPONSE_SCHEMA,
            options={"temperature": 0, "num_ctx": num_ctx, "num_predict": num_predict},
            keep_alive="30m",
            stream=False,
        )
    except ollama.ResponseError as exc:
        detail = str(exc)
        lowered = detail.lower()
        if getattr(exc, "status_code", None) == 404 or "not found" in lowered:
            raise RuntimeError(
                f"Ollama model {model} was not found. It is not loaded under that name."
            ) from exc
        if any(token in lowered for token in ("memory", "out of memory", "cannot allocate", "metal")):
            raise RuntimeError(f"Ollama ran out of memory while running {model}: {detail}") from exc
        raise RuntimeError(f"Ollama returned an error: {detail}") from exc
    except ollama.RequestError as exc:
        raise RuntimeError(f"Ollama rejected the request: {exc}") from exc
    except ConnectionError as exc:
        raise RuntimeError("Cannot reach Ollama. Check that it is running on the configured host.") from exc
    except (TimeoutError, httpx.TimeoutException) as exc:
        raise RuntimeError("Ollama did not finish within the timeout.") from exc
    except MemoryError as exc:
        raise RuntimeError("The process ran out of memory while sending page images.") from exc

    text = _message_text(response)
    if not text.strip():
        raise InvalidModelResponse("Ollama returned an empty response.")
    return text, usage_from_response(response)


def extract_features_with_vlm(
    pages: list[RenderedPage],
    *,
    client,
    model: str,
    source_filename: str,
    max_pages: int,
    max_image_bytes: int,
    num_ctx: int,
    num_predict: int,
) -> Outcome:
    """Send one document's pages, in batches when needed, and validate one result."""
    outcome = Outcome()
    try:
        groups = batch_pages(pages, max_pages, max_image_bytes)
        if not groups:
            raise RuntimeError("PDF produced no page images.")
        last_error: Exception | None = None
        for attempt in range(2):
            parts: list[dict] = []
            try:
                def collect(group: list[RenderedPage]) -> None:
                    prompt = build_extraction_prompt([page.number for page in group], len(pages))
                    started = time.perf_counter()
                    try:
                        raw, usage = ask_vlm(
                            client,
                            model,
                            prompt,
                            [page.jpeg for page in group],
                            num_ctx=num_ctx,
                            num_predict=num_predict,
                        )
                    except RuntimeError as exc:
                        outcome.request_time += time.perf_counter() - started
                        if len(group) > 1 and "context" in str(exc).lower():
                            middle = max(1, len(group) // 2)
                            logger.warning(
                                "Pages %s exceed the context window; splitting the batch.",
                                ", ".join(str(page.number) for page in group),
                            )
                            collect(group[:middle])
                            collect(group[middle:])
                            return
                        raise
                    else:
                        outcome.request_time += time.perf_counter() - started
                    outcome.usage.add(usage)
                    parse_started = time.perf_counter()
                    parts.append(parse_model_response(raw))
                    outcome.parse_time += time.perf_counter() - parse_started

                for group in groups:
                    collect(group)
                parse_started = time.perf_counter()
                merged = merge_observations(parts)
                outcome.features = validate_features(merged, source_filename)
                outcome.parse_time += time.perf_counter() - parse_started
                return outcome
            except InvalidModelResponse as exc:
                last_error = exc
                if attempt == 0:
                    logger.warning("Model JSON was invalid, retrying %s", source_filename)
                    continue
                raise ExtractionFailure(str(exc), outcome) from exc
        raise ExtractionFailure(str(last_error), outcome)
    except ExtractionFailure:
        raise
    except Exception as exc:
        raise ExtractionFailure(str(exc), outcome) from exc


def save_features(path: Path, features: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(features, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def features_path(source: Path, output_dir: Path) -> Path:
    return output_dir / f"{source.stem}.features.json"


def already_extracted(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(parsed, dict)


def append_benchmark(path: Path, row: BenchmarkRow) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "filename": row.filename,
        "pages": row.pages,
        "pdf_open_time": f"{row.pdf_open_time:.4f}",
        "rendering_time": f"{row.rendering_time:.4f}",
        "image_bytes": row.image_bytes,
        "llm_request_time": f"{row.llm_request_time:.4f}",
        "parsing_time": f"{row.parsing_time:.4f}",
        "total_time": f"{row.total_time:.4f}",
        "input_tokens": "" if row.input_tokens is None else row.input_tokens,
        "output_tokens": "" if row.output_tokens is None else row.output_tokens,
        "prompt_eval_time": "" if row.prompt_eval_time is None else f"{row.prompt_eval_time:.4f}",
        "generation_time": "" if row.generation_time is None else f"{row.generation_time:.4f}",
        "status": row.status,
        "error": row.error,
    }
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BENCHMARK_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(record)


def collect_pdfs(inputs: list[Path]) -> list[Path]:
    pdfs: list[Path] = []
    for item in inputs:
        if item.is_dir():
            found = sorted(path for path in item.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")
            if not found:
                print(f"No PDFs in {item}", file=sys.stderr)
            pdfs.extend(found)
        elif item.is_file() and item.suffix.lower() == ".pdf":
            pdfs.append(item)
        else:
            print(f"Skipping {item}: not a PDF or folder", file=sys.stderr)
    return pdfs


def _names(features: dict) -> list[str]:
    names = []
    for person in features.get("passengers") or []:
        if isinstance(person, dict) and person.get("name"):
            names.append(str(person["name"]))
    return names


def process_document(
    source: Path,
    *,
    client,
    model: str,
    dpi: float,
    jpeg_quality: int,
    max_pages: int,
    max_image_bytes: int,
    num_ctx: int,
    num_predict: int,
    destination: Path,
    benchmark: Path,
) -> None:
    started = time.perf_counter()
    row = BenchmarkRow(filename=source.name)
    pages: list[RenderedPage] = []
    print("=" * 40)
    print(f"Processing: {source.name}")
    print("=" * 40)
    print()
    try:
        if dpi > 200:
            logger.warning("DPI %s is above the 150-200 starting range and uses more memory.", dpi)
        pages, row.pdf_open_time, row.rendering_time = render_pdf_pages(source, dpi=dpi, jpeg_quality=jpeg_quality)
        row.pages = len(pages)
        row.image_bytes = sum(len(page.jpeg) for page in pages)
        groups = batch_pages(pages, max_pages, max_image_bytes)
        print(f"PDF pages: {row.pages}")
        print(f"Rendering completed: {row.rendering_time:.2f} seconds")
        if len(groups) > 1:
            print(f"Page batches: {len(groups)}")
        print(f"Asking {model}...")
        sys.stdout.flush()
        outcome = extract_features_with_vlm(
            pages,
            client=client,
            model=model,
            source_filename=source.name,
            max_pages=max_pages,
            max_image_bytes=max_image_bytes,
            num_ctx=num_ctx,
            num_predict=num_predict,
        )
        row.llm_request_time = outcome.request_time
        row.parsing_time = outcome.parse_time
        row.input_tokens = outcome.usage.input_tokens
        row.output_tokens = outcome.usage.output_tokens
        row.prompt_eval_time = outcome.usage.prompt_eval_time
        row.generation_time = outcome.usage.generation_time
        features = outcome.features or {}
        save_features(destination, features)
        row.status = "success"
        row.total_time = time.perf_counter() - started
        print()
        print(f"Document type: {features.get('document_type') or 'unknown'}")
        print("Passengers: " + (", ".join(_names(features)) or "none"))
        references = features.get("booking_references") or []
        print("References: " + (", ".join(str(item) for item in references) or "none"))
        print()
        print(f"LLM request time: {row.llm_request_time:.2f} seconds")
        print("JSON validation: successful")
        print(f"Total processing time: {row.total_time:.2f} seconds")
        print()
        print(f"Saved: {destination}")
    except Exception as exc:
        if isinstance(exc, ExtractionFailure):
            row.llm_request_time = exc.outcome.request_time
            row.parsing_time = exc.outcome.parse_time
            row.input_tokens = exc.outcome.usage.input_tokens
            row.output_tokens = exc.outcome.usage.output_tokens
            row.prompt_eval_time = exc.outcome.usage.prompt_eval_time
            row.generation_time = exc.outcome.usage.generation_time
        row.total_time = time.perf_counter() - started
        row.error = str(exc)
        row.status = "failure"
        print()
        print("Status: failed")
        print(f"Error: {exc}")
        print(f"Total processing time: {row.total_time:.2f} seconds")
        logger.error("Failed %s: %s", source, exc)
        raise
    finally:
        append_benchmark(benchmark, row)
        pages.clear()


def output_path(source: Path, output_dir: Path, used: set[Path]) -> Path:
    candidate = features_path(source, output_dir)
    suffix = 2
    while candidate in used:
        candidate = output_dir / f"{source.stem}_{suffix}.features.json"
        suffix += 1
    used.add(candidate)
    return candidate


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract travel features from PDF page images with qwen2.5vl:7b."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="PDF files or folders of PDFs")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=f"Output .json file, or a folder for several PDFs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama host (default: {DEFAULT_HOST})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama vision model (default: {DEFAULT_MODEL})")
    parser.add_argument("--timeout", type=float, default=300, help="Seconds to wait for each model response (default: 300)")
    parser.add_argument("--dpi", type=float, default=DEFAULT_DPI, help=f"PDF render DPI (default: {DEFAULT_DPI})")
    parser.add_argument("--jpeg-quality", type=int, default=75, help="JPEG quality from 1 to 100 (default: 75)")
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES, help=f"Pages sent together before batching (default: {DEFAULT_MAX_PAGES})")
    parser.add_argument(
        "--max-image-bytes",
        type=int,
        default=DEFAULT_MAX_IMAGE_BYTES,
        help=f"Image payload sent together before batching (default: {DEFAULT_MAX_IMAGE_BYTES})",
    )
    parser.add_argument("--num-ctx", type=int, default=16384, help="Ollama context length (default: 16384)")
    parser.add_argument("--num-predict", type=int, default=2048, help="Maximum tokens to generate (default: 2048)")
    parser.add_argument("--benchmark", type=Path, help="Benchmark CSV path. Default: benchmark.csv in the output folder")
    parser.add_argument("--force", action="store_true", help="Extract again even when a feature JSON already exists")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, client=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    args = parse_args(argv)
    pdfs = collect_pdfs(args.inputs)
    if not pdfs:
        print("No PDF files to process.", file=sys.stderr)
        return 1

    output = args.output if args.output is not None else DEFAULT_OUTPUT_DIR
    output_is_file = output.suffix.lower() == ".json"
    if output_is_file and len(pdfs) != 1:
        print("Pass a folder to --output when processing more than one PDF.", file=sys.stderr)
        return 1
    if output_is_file:
        output.parent.mkdir(parents=True, exist_ok=True)
        benchmark = args.benchmark or output.with_name("benchmark.csv")
    else:
        output.mkdir(parents=True, exist_ok=True)
        benchmark = args.benchmark or (output / "benchmark.csv")

    used: set[Path] = set()
    pending: list[tuple[Path, Path]] = []
    for source in pdfs:
        destination = output if output_is_file else output_path(source, output, used)
        if not args.force and already_extracted(destination):
            print(f"{source.name}: already extracted -> {destination}")
            continue
        pending.append((source, destination))

    if not pending:
        print("All PDF files already have feature JSON.")
        return 0

    ollama_client = client if client is not None else build_client(args.host, args.timeout)
    failures = 0
    for source, destination in pending:
        try:
            process_document(
                source,
                client=ollama_client,
                model=args.model,
                dpi=args.dpi,
                jpeg_quality=args.jpeg_quality,
                max_pages=args.max_pages,
                max_image_bytes=args.max_image_bytes,
                num_ctx=args.num_ctx,
                num_predict=args.num_predict,
                destination=destination,
                benchmark=benchmark,
            )
        except Exception:
            failures += 1
            continue
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
