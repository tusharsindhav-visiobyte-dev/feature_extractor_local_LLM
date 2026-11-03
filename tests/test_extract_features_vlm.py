"""Tests for the direct PDF vision pipeline. They do not call Ollama."""

from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from data_collection import feature_row
from extract_features_vlm import (
    BENCHMARK_COLUMNS,
    RenderedPage,
    batch_pages,
    main,
    nanoseconds_to_seconds,
    render_pdf_pages,
    validate_features,
)
from travel_schema import TravelDocument


def _write_pdf(path: Path, pages: list[str]) -> None:
    import pymupdf

    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def _train_payload(**overrides) -> dict:
    payload = {
        "document_type": "train",
        "document_subtype": "",
        "document_title": "E-Ticket",
        "booking_references": ["AY5811803598"],
        "confirmation_codes": [],
        "ticket_numbers": [],
        "booking_status": "booked",
        "booking_date": "18 Apr 2025",
        "issue_date": "",
        "travelers": [{"name": "Chandrakant Rana", "role": "", "ticket_number": "", "seat": "S1", "coach": ""}],
        "transport": [
            {
                "mode": "train",
                "carrier": "IRCTC",
                "service_number": "12902",
                "from": {"name": "Ahmedabad", "code": "", "terminal": "", "gate": "", "datetime": "22 Apr 2025 22:15"},
                "to": {"name": "Dwarka", "code": "", "terminal": "", "gate": "", "datetime": "23 Apr 2025 06:50"},
                "seat": "S1",
                "coach": "",
                "cabin": "Sleeper",
                "platform": "2",
                "boarding_time": "",
                "status": "",
                "fare": "INR 860",
            }
        ],
        "lodging": [],
        "car_rentals": [],
        "emails": [],
        "phones": ["040-61656789"],
        "baggage": "",
        "notes": ["Carry a photo id"],
        "currency": "INR",
        "price_total": "INR 860",
        "payment_status": "paid",
    }
    payload.update(overrides)
    return payload


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.seen_images = []

    def chat(self, **kwargs):
        self.calls += 1
        images = kwargs["messages"][1]["images"]
        self.seen_images.append(list(images))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        content = item if isinstance(item, str) else json.dumps(item)
        return SimpleNamespace(
            message=SimpleNamespace(content=content),
            prompt_eval_count=11,
            eval_count=22,
            prompt_eval_duration=2_000_000_000,
            eval_duration=3_000_000_000,
            total_duration=5_000_000_000,
        )


class VisionSchemaTests(unittest.TestCase):
    def test_train_payload_matches_the_travel_document(self):
        features = validate_features(_train_payload(), "AY5811803598.pdf")
        self.assertEqual(features["document_type"], "train")
        self.assertEqual(features["passengers"][0]["name"], "Chandrakant Rana")
        self.assertEqual(features["trains"][0]["train_number"], "12902")
        self.assertEqual(features["trains"][0]["departure"]["iso"], "2025-04-22T22:15")
        self.assertEqual(features["trains"][0]["platform"], "2")
        self.assertEqual(features["booking_date"]["iso"], "2025-04-18")
        self.assertEqual(features["total_price"]["amount"], "860")
        self.assertEqual(features["provider_contact"]["phones"], ["040-61656789"])
        self.assertNotIn("prompt_eval_count", features)
        TravelDocument.model_validate(features)
        row = feature_row(Path("AY5811803598.features.json"), features)
        self.assertEqual(row["document_type"], "train")
        self.assertIn("Chandrakant Rana", row["passengers.name"])

    def test_flat_places_and_contact_names_are_normalized(self):
        features = validate_features(
            {
                "document_type": "bus",
                "travelers": [
                    {"name": "Chandrakant Rana"},
                    {"name": "9876543210"},
                    {"name": "help@example.com"},
                ],
                "transport": [
                    {
                        "mode": "bus",
                        "carrier": "AbhiBus",
                        "service_number": "120896",
                        "from_place": "Ahmedabad",
                        "from_datetime": "22 Apr 2025 22:15",
                        "to_place": "Dwarka",
                        "to_datetime": "23 Apr 2025 06:50",
                    }
                ],
            },
            "ticket.pdf",
        )
        self.assertEqual(features["document_type"], "bus")
        self.assertEqual([person["name"] for person in features["passengers"]], ["Chandrakant Rana"])
        self.assertEqual(features["buses"][0]["departure_location"], "Ahmedabad")
        self.assertEqual(features["buses"][0]["arrival"]["iso"], "2025-04-23T06:50")

    def test_unknown_type_is_preserved_when_no_segment_is_visible(self):
        features = validate_features({"document_type": "unknown"}, "blank.pdf")
        self.assertEqual(features["document_type"], "unknown")
        self.assertEqual(features["extraction_status"], "unknown")
        self.assertEqual(features["flights"], [])

    def test_train_and_bus_pages_become_an_itinerary(self):
        from extract_features_vlm import merge_observations

        merged = merge_observations(
            [
                _train_payload(),
                {
                    "document_type": "bus",
                    "travelers": [{"name": "Chandrakant Rana", "seat": "S1"}],
                    "transport": [
                        {
                            "mode": "bus",
                            "carrier": "AbhiBus",
                            "service_number": "120896",
                            "from": {"name": "Ahmedabad", "datetime": "24 Apr 2025 08:00"},
                            "to": {"name": "Dwarka", "datetime": "24 Apr 2025 14:00"},
                        }
                    ],
                },
            ]
        )
        features = validate_features(merged, "mixed.pdf")
        self.assertEqual(features["document_type"], "itinerary")
        self.assertEqual(len(features["trains"]), 1)
        self.assertEqual(len(features["buses"]), 1)
        self.assertEqual(len(features["passengers"]), 1)

    def test_saved_ocr_feature_file_still_loads_for_analysis(self):
        path = Path("data/APPLE_OCR+QWEN/AY5811803598.features.json")
        if not path.is_file():
            self.skipTest("sample feature file is not in this checkout")
        payload = json.loads(path.read_text(encoding="utf-8"))
        document = TravelDocument.model_validate(payload)
        row = feature_row(path, payload)
        self.assertEqual(row["document_type"], document.document_type.value)
        self.assertEqual(row["source_file"], path.name)


class PdfRenderingTests(unittest.TestCase):
    def test_pages_stay_in_order_and_are_jpeg(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ticket.pdf"
            _write_pdf(path, ["page one", "page two"])
            pages, open_time, render_time = render_pdf_pages(path, dpi=150, jpeg_quality=70)
        self.assertEqual([page.number for page in pages], [1, 2])
        self.assertTrue(all(page.jpeg.startswith(b"\xff\xd8") for page in pages))
        self.assertGreater(open_time, 0)
        self.assertGreater(render_time, 0)

    def test_password_and_corrupt_pdfs_fail_clearly(self):
        import pymupdf
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            protected = folder / "locked.pdf"
            document = pymupdf.open()
            document.new_page().insert_text((72, 72), "secret ticket")
            document.save(
                protected,
                encryption=pymupdf.PDF_ENCRYPT_AES_256,
                user_pw="secret",
                owner_pw="secret",
            )
            document.close()
            with self.assertRaises(RuntimeError) as locked:
                render_pdf_pages(protected, dpi=150, jpeg_quality=70)
            self.assertIn("password-protected", str(locked.exception))

            corrupt = folder / "broken.pdf"
            corrupt.write_bytes(b"this is not a pdf")
            with self.assertRaises(RuntimeError) as broken:
                render_pdf_pages(corrupt, dpi=150, jpeg_quality=70)
            self.assertIn("Could not open PDF", str(broken.exception))

    def test_batching_keeps_every_page_in_order(self):
        pages = [RenderedPage(number=number, jpeg=b"x" * 100) for number in range(1, 6)]
        groups = batch_pages(pages, max_pages=2, max_image_bytes=10_000)
        self.assertEqual([[page.number for page in group] for group in groups], [[1, 2], [3, 4], [5]])
        oversized = [RenderedPage(number=1, jpeg=b"x" * 500)]
        self.assertEqual(batch_pages(oversized, max_pages=4, max_image_bytes=10)[0][0].number, 1)
        split = batch_pages(pages[:2], max_pages=4, max_image_bytes=150)
        self.assertEqual([[page.number for page in group] for group in split], [[1], [2]])


class PipelineTests(unittest.TestCase):
    def _run(self, folder: Path, names: list[str], client: FakeClient, extra: list[str] | None = None) -> int:
        paths = []
        for name in names:
            path = folder / name
            _write_pdf(path, [name])
            paths.append(str(path))
        output = folder / "QWEN_VL"
        benchmark = folder / "benchmark.csv"
        return main(
            [*paths, "-o", str(output), "--benchmark", str(benchmark), "--dpi", "72", *(extra or [])],
            client=client,
        )

    def test_one_pdf_writes_schema_json_and_benchmark(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            client = FakeClient([_train_payload()])
            code = self._run(folder, ["8352727415.pdf"], client)
            self.assertEqual(code, 0)
            self.assertEqual(client.calls, 1)
            self.assertTrue(client.seen_images[0][0].startswith(b"\xff\xd8"))
            saved = json.loads((folder / "QWEN_VL" / "8352727415.features.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["document_type"], "train")
            self.assertEqual(saved["source_filename"], "8352727415.pdf")
            TravelDocument.model_validate(saved)
            with (folder / "benchmark.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(list(rows[0]), list(BENCHMARK_COLUMNS))
            self.assertEqual(rows[0]["filename"], "8352727415.pdf")
            self.assertEqual(rows[0]["pages"], "1")
            self.assertEqual(rows[0]["status"], "success")
            self.assertEqual(rows[0]["input_tokens"], "11")
            self.assertEqual(rows[0]["prompt_eval_time"], "2.0000")
            self.assertEqual(rows[0]["generation_time"], "3.0000")
            self.assertNotIn("llm_request_time", saved)

    def test_invalid_json_is_retried_then_a_later_pdf_still_runs(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            client = FakeClient(["not json", _train_payload(), RuntimeError("Cannot reach Ollama.")])
            code = self._run(folder, ["first.pdf", "second.pdf"], client)
            self.assertEqual(code, 1)
            self.assertEqual(client.calls, 3)
            self.assertTrue((folder / "QWEN_VL" / "first.features.json").is_file())
            self.assertFalse((folder / "QWEN_VL" / "second.features.json").exists())
            with (folder / "benchmark.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["status"] for row in rows], ["success", "failure"])
            self.assertIn("Cannot reach Ollama", rows[1]["error"])

    def test_directory_argument_and_existing_output_is_skipped(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "docs"
            source.mkdir()
            _write_pdf(source / "a.pdf", ["a"])
            _write_pdf(source / "b.pdf", ["b"])
            output = folder / "out"
            output.mkdir()
            (output / "a.features.json").write_text("{}\n", encoding="utf-8")
            client = FakeClient([_train_payload()])
            code = main([str(source), "-o", str(output), "--dpi", "72"], client=client)
            self.assertEqual(code, 0)
            self.assertEqual(client.calls, 1)
            saved = json.loads((output / "b.features.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["document_type"], "train")

    def test_context_overflow_splits_pages_instead_of_dropping_them(self):
        import tempfile

        class SplittingClient:
            def __init__(self):
                self.sizes = []

            def chat(self, **kwargs):
                images = kwargs["messages"][1]["images"]
                self.sizes.append(len(images))
                if len(images) > 1:
                    raise RuntimeError("request exceeds the available context size")
                payload = _train_payload() if len(self.sizes) == 2 else {
                    "document_type": "bus",
                    "travelers": [{"name": "Chandrakant Rana"}],
                    "transport": [
                        {
                            "mode": "bus",
                            "carrier": "AbhiBus",
                            "service_number": "120896",
                            "from": {"name": "Ahmedabad", "datetime": "24 Apr 2025 08:00"},
                            "to": {"name": "Dwarka", "datetime": "24 Apr 2025 14:00"},
                        }
                    ],
                }
                return SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(payload)),
                    prompt_eval_count=4,
                    eval_count=5,
                    prompt_eval_duration=1_000_000_000,
                    eval_duration=1_000_000_000,
                    total_duration=2_000_000_000,
                )

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            path = folder / "wide.pdf"
            _write_pdf(path, ["page one", "page two"])
            client = SplittingClient()
            code = main([str(path), "-o", str(folder / "out"), "--dpi", "72", "--num-ctx", "8192"], client=client)
            self.assertEqual(code, 0)
            self.assertEqual(client.sizes, [2, 1, 1])
            saved = json.loads((folder / "out" / "wide.features.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["document_type"], "itinerary")
            self.assertEqual(len(saved["trains"]), 1)
            self.assertEqual(len(saved["buses"]), 1)

    def test_nanoseconds_are_reported_in_seconds(self):
        self.assertEqual(nanoseconds_to_seconds(1_500_000_000), 1.5)
        self.assertIsNone(nanoseconds_to_seconds(None))


if __name__ == "__main__":
    unittest.main()
