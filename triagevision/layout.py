"""The tag's internal layout, used as a hard geometric constraint.

Printed triage tags carry three elements stacked in a fixed order:

    +-----------------------------+
    |        IMMEDIATE            |  <- banner: the acuity word
    |        EA1568511            |  <- the same ID as plain text
    |     |||| ||| |||| ||||      |  <- the ID again, as a barcode
    +-----------------------------+

That order never varies, and it is rigid *relative to the tag* -- so if one tag
in a photo is rotated and its neighbour is not, each tag's own three elements
stay aligned with each other. Orientation is therefore a per-tag property that
can be recovered from the tag itself, with no reference to the frame, the other
tags, or the field color.

This module turns that invariant into two concrete abilities:

  1. Find the actual text rows, instead of assuming the banner occupies some
     fixed fraction of the tag's height. Vendors differ; ink does not.
  2. Settle which way is up. The barcode sits at one end and the banner at the
     other, so the side of the symbol carrying the text rows is "up".

Because the printed ID repeats the barcode payload, the resulting orientation is
verifiable rather than assumed -- see `TriageTagDetector._resolve_stack`.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TextRow:
    """A horizontal band of ink inside an upright tag crop."""

    y0: int
    y1: int
    density: float

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def center(self) -> float:
        return (self.y0 + self.y1) / 2.0


@dataclass
class Stack:
    """The resolved layout of one tag crop."""

    banner: TextRow | None
    ident: TextRow | None
    flipped: bool          # True if the crop must be rotated 180 to be upright
    rows: list[TextRow]


def ink_profile(crop: np.ndarray) -> np.ndarray:
    """Per-row fraction of 'ink' pixels, polarity-independent.

    Uses gradient magnitude rather than a threshold on brightness, because the
    banner is light-on-dark while the ID line is dark-on-light and a single
    binarization cannot serve both. Edges appear for either polarity.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    mag = cv2.convertScaleAbs(gx)
    _, edges = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (edges > 0).mean(axis=1)


def find_text_rows(
    crop: np.ndarray, min_density: float = 0.06, min_height_frac: float = 0.035
) -> list[TextRow]:
    """Group rows of ink into bands, largest-to-smallest by prominence."""
    if crop is None or crop.size == 0:
        return []
    profile = ink_profile(crop)
    h = len(profile)
    if h < 8:
        return []

    # Smooth so the gaps between glyphs do not fragment a single line of text.
    k = max(3, int(h * 0.02) | 1)
    smooth = np.convolve(profile, np.ones(k) / k, mode="same")

    active = smooth > max(min_density, float(smooth.max()) * 0.28)
    min_h = max(3, int(h * min_height_frac))

    rows: list[TextRow] = []
    start: int | None = None
    for i, on in enumerate(active):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start >= min_h:
                rows.append(TextRow(start, i, float(smooth[start:i].mean())))
            start = None
    if start is not None and h - start >= min_h:
        rows.append(TextRow(start, h, float(smooth[start:h].mean())))

    return rows


def resolve_stack(crop: np.ndarray, barcode_center_y: float) -> Stack:
    """Work out which end of a symmetric crop holds the banner.

    `crop` is expected to be centred on the barcode, extending far enough in
    both directions to contain the tag whichever way it faces. The tag body is
    on whichever side of the symbol actually carries text.
    """
    rows = find_text_rows(crop)
    if not rows:
        return Stack(None, None, False, [])

    # Ignore the barcode's own band: it is the densest thing in the crop and
    # sits at the seed point.
    body = [
        r
        for r in rows
        if not (r.y0 - 2 <= barcode_center_y <= r.y1 + 2)
    ]
    above = [r for r in body if r.center < barcode_center_y]
    below = [r for r in body if r.center > barcode_center_y]

    # Score each side by how much text it holds close to the symbol. The tag
    # body is compact and adjacent; the far side is background or a neighbour.
    def weight(rows_side: list[TextRow]) -> float:
        return sum(
            r.density * r.height / (1.0 + abs(r.center - barcode_center_y))
            for r in rows_side
        )

    # Hysteresis. Callers hand us a crop that layout priors already believe is
    # banner-up, so this is a veto, not a vote: only overturn that when the far
    # side holds clearly more text. Flipping on a narrow margin turns a readable
    # tag into an unreadable one, which is a worse failure than doing nothing.
    flipped = weight(below) > weight(above) * 1.6
    side = below if flipped else above

    if not side:
        return Stack(None, None, flipped, rows)

    # Nearest the barcode is the printed ID; furthest is the banner.
    side.sort(key=lambda r: abs(r.center - barcode_center_y))
    ident = side[0]
    banner = side[-1] if len(side) > 1 else None
    return Stack(banner=banner, ident=ident, flipped=flipped, rows=rows)


def flip_row(row: TextRow, height: int) -> TextRow:
    """Re-express a row's bounds after the crop is rotated 180 degrees."""
    return TextRow(height - row.y1, height - row.y0, row.density)


def pad_row(row: TextRow, height: int, frac: float = 0.35) -> tuple[int, int]:
    """Widen a row's bounds a little so ascenders and descenders survive."""
    pad = max(2, int(row.height * frac))
    return max(0, row.y0 - pad), min(height, row.y1 + pad)
