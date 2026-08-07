"""Command line entry point: `python -m triagevision.cli IMAGE [IMAGE ...]`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import DetectorConfig
from .detector import TriageTagDetector, annotate, load_image


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="triagevision",
        description="Read patient ID and acuity from photographs of triage tags.",
    )
    p.add_argument("images", nargs="+", type=Path, help="image files to process")
    p.add_argument("--config", type=Path, help="JSON file of DetectorConfig overrides")
    p.add_argument(
        "--use-color",
        action="store_true",
        help="enable color-field segmentation (off by default; it degrades "
        "accuracy under colored lighting -- see README)",
    )
    p.add_argument(
        "--require-text",
        action="store_true",
        help="report UNKNOWN rather than guessing acuity from color",
    )
    p.add_argument("--ocr", default=None, help="OCR backend: auto|tesseract|none")
    p.add_argument(
        "--roster",
        action="store_true",
        help="print only patient_id/acuity pairs instead of full detail",
    )
    p.add_argument(
        "--annotate",
        type=Path,
        metavar="DIR",
        help="also write annotated copies of each image to DIR",
    )
    p.add_argument("--workers", type=int, default=None, help="reader threads per image")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = DetectorConfig.from_json(args.config) if args.config else DetectorConfig()
    if args.use_color:
        cfg.use_color = True
    if args.require_text:
        cfg.require_text = True
    if args.ocr:
        cfg.ocr_backend = args.ocr
    if args.workers is not None:
        cfg.max_workers = args.workers

    detector = TriageTagDetector(cfg)
    if not detector.reader.available:
        print(
            "warning: no OCR backend available; acuity will fall back to field "
            "color. Install tesseract and pytesseract.",
            file=sys.stderr,
        )

    if args.annotate:
        args.annotate.mkdir(parents=True, exist_ok=True)

    exit_code = 0
    payload = []
    for path in args.images:
        try:
            result = detector.detect(path)
        except Exception as exc:  # keep going through a batch
            print(f"error: {path}: {exc}", file=sys.stderr)
            exit_code = 1
            continue

        entry = {"image": str(path)}
        entry.update(
            {"tags": result.roster()} if args.roster else result.to_dict()
        )
        payload.append(entry)

        if args.annotate:
            import cv2

            out = args.annotate / f"{path.stem}_annotated.png"
            cv2.imwrite(str(out), annotate(load_image(path), result))

    json.dump(payload if len(payload) != 1 else payload[0], sys.stdout, indent=2)
    sys.stdout.write("\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
