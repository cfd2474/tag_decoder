"""Fast readability check, so an unusable photo can be rejected immediately.

Poor photographs are expected in the field, and processing one takes several
seconds to arrive at a bad answer. This runs in a few hundred milliseconds and
answers a narrower question: is there anything in this frame worth reading?

It is designed to run CONCURRENTLY with the main pipeline. A good frame is
never delayed by it -- the check finishes long before the barcode stage does,
and the caller simply carries on. Only a frame that fails is stopped, and by
then the expensive per-tag OCR has not started.

Judged on evidence, not on a picture-quality proxy. Generic sharpness metrics
were tested and rejected: one sheet still read 15/15 after being blurred well
below the sharpness of another that read 7/15. What the check actually does is
try to find something -- a symbol, then a symbol at an angle, then a printed
word -- and rate the frame by what turns up.

  ok         a barcode decoded on the fast path; the frame is workable
  degraded   nothing decoded, but banner words are present, so at best the
             patient IDs will come from OCR and some tags may be missing
  unusable   neither symbols nor words; there is nothing here to read
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from . import barcode as barcode_mod
from . import textfind

# Angle for the second probe. A frame lying diagonally decodes nothing on a
# straight pass -- measured, a sheet rotated 37 degrees drops from nine symbols
# to zero -- and rejecting it as unreadable would be badly wrong, because the
# full pipeline's rotation sweep recovers it. One cheap rotated probe tells a
# diagonal frame apart from an unreadable one.
PROBE_ANGLE = 45.0

# Banner scan for the last-resort probe. Deliberately small and single-pass:
# it only has to establish that printed words exist somewhere.
WORD_SCAN_WIDTH = 1100


@dataclass
class Preflight:
    """Verdict from the fast readability check."""

    rating: str                 # "ok" | "degraded" | "unusable"
    symbols_found: int
    words_found: int
    sharpness: float
    elapsed_ms: float
    advice: str | None = None

    @property
    def usable(self) -> bool:
        return self.rating == "ok"

    def should_abort(self, abort_on: str) -> bool:
        """Whether the caller's policy says to stop before the expensive work."""
        if abort_on == "never":
            return False
        if abort_on == "unusable":
            return self.rating == "unusable"
        return self.rating in ("unusable", "degraded")   # "degraded"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rating": self.rating,
            "symbols_found": self.symbols_found,
            "words_found": self.words_found,
            "sharpness": round(self.sharpness, 1),
            "elapsed_ms": round(self.elapsed_ms, 1),
            "advice": self.advice,
        }


_ADVICE = {
    "degraded": (
        "image quality poor: no barcode could be decoded on a fast pass, so "
        "patient IDs would be read by OCR at best and some tags may be missed "
        "entirely. Retake: hold the camera steady, fill the frame with the "
        "tags, and keep glare off the barcodes"
    ),
    "unusable": (
        "no triage tags could be found in this image at all -- no barcode and "
        "no readable banner text. Retake with the tags clearly visible and "
        "filling more of the frame"
    ),
}


def check(image: np.ndarray, keywords, workers: int = 4) -> Preflight:
    """Rate a frame's readability in a few hundred milliseconds.

    Escalates only on bad news, so the common case stays fast: a frame whose
    symbols decode straight away costs one pass and returns.
    """
    started = time.perf_counter()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # Cheap and informational; deliberately NOT the basis of the verdict.
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    sharpness = float(cv2.Laplacian(small, cv2.CV_64F).var())

    symbols = len(barcode_mod._decode_pass(gray, 1.0, barcode_mod.DEFAULT_FORMATS))

    if symbols == 0:
        # Diagonal frame, or genuinely unreadable? One rotated probe decides.
        symbols = len(
            barcode_mod._rotate_pass(small, PROBE_ANGLE, barcode_mod.DEFAULT_FORMATS)
        )

    words = 0
    if symbols == 0:
        words = len(
            textfind.find_banners(
                image, keywords, scan_width=WORD_SCAN_WIDTH,
                psms=(11,), workers=workers, probe_only=True,
            )
        )

    if symbols > 0:
        rating = "ok"
    elif words > 0:
        rating = "degraded"
    else:
        rating = "unusable"

    return Preflight(
        rating=rating,
        symbols_found=symbols,
        words_found=words,
        sharpness=sharpness,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        advice=_ADVICE.get(rating),
    )
