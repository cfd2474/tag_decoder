"""Finding tags by their printed banner word -- a localizer, not just a reader.

The barcode is the better ID source when it decodes, but it is a brittle
localizer: a slightly soft photo can leave most symbols undecodable while the
printed words remain perfectly legible. Measured on one such frame, 7 of 15
barcodes decoded while all 15 banner words could be read. With the barcode as
the sole localizer, those 8 patients were simply absent from the output.

Wide letter strokes survive blur that destroys narrow bars. That asymmetry is
the whole reason this module exists: scanning the frame for the six vocabulary
words finds tags that the decoder cannot, and the printed ID line sitting
directly beneath each banner supplies an ID for them.

IDs recovered this way are marked `ocr_only` and flagged. They are a fallback,
not a promotion: OCR on an open-vocabulary serial is exactly the error-prone
case the barcode exists to avoid. When a barcode is present it still wins.
"""

from __future__ import annotations

import difflib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

from .types import Acuity, BBox

_ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_ALNUM = _ALPHA + "0123456789"


@dataclass
class BannerHit:
    """A vocabulary word located in the frame, anchoring one tag."""

    acuity: Acuity
    text: str
    bbox: BBox            # word box, full-resolution coordinates
    score: float
    margin: float
    conf: int             # engine's own word confidence, 0-100


def _config(psm: int, whitelist: str) -> str:
    return (
        f"--oem 1 --psm {psm} -c tessedit_char_whitelist={whitelist} "
        "-c load_system_dawg=0 -c load_freq_dawg=0"
    )


def _scan_variants(gray: np.ndarray):
    """Binarizations to scan. Banners come in both polarities, and the plain
    greyscale often beats both, so all three are tried.
    """
    yield gray
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield otsu
    yield cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def _plausible_word_box(word: str, w: int, h: int) -> bool:
    """Reject boxes that cannot be a single banner word on a single tag.

    The engine sometimes merges text across neighbouring tags into one box --
    an `IMMEDIATE` 1635px wide spanning three tags. Anchoring on that would put
    the ID band across several tags and produce a fabricated ID. A word's box
    should be about 0.62 x its character count in aspect; anything far wider is
    a merge.
    """
    if h <= 0 or w <= 0:
        return False
    aspect = w / float(h)
    expected = 0.62 * max(len(word), 3)
    return 0.45 * expected <= aspect <= 1.9 * expected


def _match(token: str, keywords: dict[str, Acuity], min_ratio: float, min_margin: float):
    """Fuzzy-match a token to the vocabulary, using the same margin rule the
    banner reader uses: the lead over the runner-up, not the raw score.
    """
    ratios = {w: difflib.SequenceMatcher(None, token, w).ratio() for w in keywords}
    best_word = max(ratios, key=ratios.get)
    best = ratios[best_word]
    runner_up = max((v for w, v in ratios.items() if w != best_word), default=0.0)
    margin = best - runner_up
    if best < min_ratio or margin < min_margin:
        return None
    return keywords[best_word], best, margin


def find_banners(
    image: np.ndarray,
    keywords: dict[str, Acuity],
    scan_width: int = 1800,
    psms: tuple[int, ...] = (6, 11),
    min_ratio: float = 0.72,
    min_margin: float = 0.20,
    workers: int = 0,
    probe_only: bool = False,
) -> list[BannerHit]:
    """Locate banner words in the frame.

    Scanning happens on a downscaled copy -- the words are large and the engine
    is far faster on fewer pixels -- and boxes are mapped back to full
    resolution. Variants and segmentation modes run in parallel, since each OCR
    call is a separate process.

    `probe_only` runs just the single cheapest combination. A sharp frame where
    the barcodes already found everything should not pay for a full scan, but
    "found some, stop looking" is not a safe rule either -- so the caller runs
    the probe first and escalates only when the probe produces evidence of a
    tag the barcodes missed.
    """
    try:
        import pytesseract
    except ImportError:  # pragma: no cover
        return []

    h, w = image.shape[:2]
    if w == 0 or h == 0:
        return []
    # Bound the LONG side, not the width. Scaling a portrait frame to a fixed
    # width leaves it taller than it was wide, and the engine's cost tracks
    # total pixels: a 1800x2400 scan measured 8.9s against 1.0s for 1800x1350.
    scale = min(1.0, scan_width / float(max(w, h)))
    small = (
        cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else image
    )
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small

    variants = list(_scan_variants(gray))
    # The probe uses sparse-text mode: it is the cheaper of the two and only
    # has to prove that a banner exists outside the located tags.
    probe_psm = 11 if 11 in psms else psms[0]
    jobs = ([(variants[0], probe_psm)] if probe_only
            else [(v, psm) for v in variants for psm in psms])

    def scan(job) -> list[BannerHit]:
        variant, psm = job
        try:
            data = pytesseract.image_to_data(
                variant,
                config=_config(psm, _ALPHA),
                output_type=pytesseract.Output.DICT,
            )
        except Exception:
            return []

        out: list[BannerHit] = []
        inv = 1.0 / scale if scale else 1.0
        for i, raw in enumerate(data["text"]):
            token = raw.strip().upper()
            if len(token) < 4:
                continue
            hit = _match(token, keywords, min_ratio, min_margin)
            if hit is None:
                continue
            acuity, score, margin = hit
            x, y = data["left"][i], data["top"][i]
            bw, bh = data["width"][i], data["height"][i]
            if not _plausible_word_box(token, bw, bh):
                continue
            try:
                conf = int(float(data["conf"][i]))
            except (TypeError, ValueError):
                conf = -1
            out.append(
                BannerHit(
                    acuity=acuity,
                    text=token,
                    bbox=BBox(int(x * inv), int(y * inv), int(bw * inv), int(bh * inv)),
                    score=score,
                    margin=margin,
                    conf=conf,
                )
            )
        return out

    n = workers if workers > 0 else min(len(jobs), 8)
    results: list[BannerHit] = []
    if n > 1:
        with ThreadPoolExecutor(max_workers=n) as pool:
            for batch in pool.map(scan, jobs):
                results.extend(batch)
    else:
        for job in jobs:
            results.extend(scan(job))

    return _dedupe(results)


def _dedupe(hits: list[BannerHit], iou_thresh: float = 0.30) -> list[BannerHit]:
    """One hit per physical banner, keeping the best-supported read.

    The same word is found by several variants at slightly different bounds;
    those are one tag. Ranked by exactness of the match first, then engine
    confidence, then box area.
    """
    ordered = sorted(
        hits, key=lambda b: (b.score, b.margin, b.conf, b.bbox.area), reverse=True
    )
    kept: list[BannerHit] = []
    for hit in ordered:
        if any(hit.bbox.iou(k.bbox) > iou_thresh for k in kept):
            continue
        if any(
            k.bbox.contains_point(hit.bbox.cx, hit.bbox.cy) for k in kept
        ):
            continue
        kept.append(hit)
    return kept


# Where the printed ID sits relative to the banner word, in multiples of the
# word's own height. The stack is banner / ID / barcode, so the ID line is
# immediately below the banner; the band is widened horizontally because the ID
# is usually a little wider than the word above it.
ID_BAND_TOP = 0.55
ID_BAND_BOTTOM = 2.10
ID_BAND_SIDE = 0.35


def id_band(hit: BannerHit, shape: tuple[int, int]) -> BBox:
    """The region beneath a banner word where the printed patient ID must be."""
    h_img, w_img = shape
    b = hit.bbox
    y0 = max(0, int(b.y + b.h * ID_BAND_TOP))
    y1 = min(h_img, int(b.y + b.h * ID_BAND_BOTTOM))
    x0 = max(0, int(b.x - b.w * ID_BAND_SIDE))
    x1 = min(w_img, int(b.x + b.w * (1.0 + ID_BAND_SIDE)))
    return BBox(x0, y0, max(0, x1 - x0), max(0, y1 - y0))


def id_shape(text: str) -> str:
    """Character-class signature of an ID: 'EA1568511' -> 'AADDDDDDD'."""
    return "".join("D" if c.isdigit() else ("A" if c.isalpha() else "?") for c in text)


def shape_template(decoded_ids: list[str]) -> str | None:
    """The dominant ID shape among symbols that decoded in this frame.

    Patient numbers are pre-printed vendor serials, so the ID pattern has to
    stay permissive globally -- the next batch's format is unknowable. But
    within ONE frame the tags come from one batch, and any barcode that decoded
    tells us that batch's shape exactly. That makes a strict local check
    possible where a strict global one would be wrong.

    This is what stops OCR fabricating IDs. Without it, garbage like
    'SSPGVEPQEB' or '28S84A' satisfies the permissive pattern and is reported as
    a patient.
    """
    shapes = [id_shape(t) for t in decoded_ids if t]
    if not shapes:
        return None
    counts: dict[str, int] = {}
    for s in shapes:
        counts[s] = counts.get(s, 0) + 1
    best = max(counts, key=counts.get)
    # Only trust it if it is genuinely the batch's shape, not a one-off.
    return best if counts[best] >= max(2, len(shapes) // 2) else None


def read_id_text(
    image: np.ndarray,
    box: BBox,
    is_valid,
    template: str | None = None,
) -> str | None:
    """OCR a printed patient ID out of `box`.

    Upscaled, because the ID line is far smaller than the banner above it, and
    tried in both polarities. When `template` is given, a candidate must match
    that character-class shape -- a fabricated ID is worse than no ID, and on a
    soft frame OCR readily produces plausible-looking junk.
    """
    try:
        import pytesseract
    except ImportError:  # pragma: no cover
        return None

    crop = image[box.y : box.y + box.h, box.x : box.x + box.w]
    if crop.size == 0 or box.w < 24 or box.h < 8:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 5, 60, 60)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    candidates: list[str] = []
    for variant in (gray, otsu, cv2.bitwise_not(otsu), clahe):
        for psm in (7, 6):
            try:
                raw = pytesseract.image_to_string(
                    variant, config=_config(psm, _ALNUM)
                )
            except Exception:
                continue
            for token in raw.replace("\n", " ").split():
                token = token.strip()
                if token and is_valid(token):
                    candidates.append(token)
        # An exact shape match is as good as this gets; stop early.
        if template and any(id_shape(t) == template for t in candidates):
            break

    if not candidates:
        return None
    if template:
        exact = [t for t in candidates if id_shape(t) == template]
        if not exact:
            return None
        # Most frequently agreed reading wins.
        return max(set(exact), key=exact.count)
    return max(candidates, key=len)
