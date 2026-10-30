#!/usr/bin/env python3
"""Extract text from PDFs with Apple OCR and save one .txt file per PDF."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from apple_ocr import extract_pdf, pages_to_text


def collect_pdfs(inputs: list[Path]) -> list[Path]:
    pdfs: list[Path] = []
    for item in inputs:
        if item.is_dir():
            found = sorted(
                path
                for path in item.iterdir()
                if path.is_file() and path.suffix.lower() == ".pdf"
            )
            if not found:
                print(f"No PDFs in {item}", file=sys.stderr)
            pdfs.extend(found)
        elif item.is_file() and item.suffix.lower() == ".pdf":
            pdfs.append(item)
        else:
            print(f"Skipping {item}: not a PDF or folder", file=sys.stderr)
    return pdfs


def output_path(pdf: Path, output_dir: Path | None, used: set[Path]) -> Path:
    directory = output_dir if output_dir is not None else pdf.parent
    candidate = directory / f"{pdf.stem}.txt"
    suffix = 2
    while candidate in used:
        candidate = directory / f"{pdf.stem}_{suffix}.txt"
        suffix += 1
    used.add(candidate)
    return candidate


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract text from PDFs with Apple Vision OCR and save .txt files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="PDF files or folders containing PDFs",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Folder for the text files. Default: next to each PDF",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="Page render scale. 2 is about 144 DPI (default: 2)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en-US"],
        help="Vision language codes, for example en-US (default: en-US)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast recognition instead of accurate",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Drop recognized lines below this confidence, from 0 to 1",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pdfs = collect_pdfs(args.inputs)
    if not pdfs:
        print("No PDF files to process.", file=sys.stderr)
        return 1

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    used: set[Path] = set()
    failures = 0
    for pdf in pdfs:
        destination = output_path(pdf, args.output_dir, used)
        try:
            pages = extract_pdf(
                pdf,
                scale=args.scale,
                languages=args.languages,
                fast=args.fast,
                min_confidence=args.min_confidence,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            failures += 1
            print(f"Failed {pdf}: {exc}", file=sys.stderr)
            continue

        destination.write_text(pages_to_text(pages), encoding="utf-8")
        print(f"{pdf.name}: {len(pages)} page(s) -> {destination}")

    return 1 if failures else 0


if __name__ == "__main__":
    # raise SystemExit(main())
    pass
