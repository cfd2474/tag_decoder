"""Color-field segmentation: find tag-shaped colored regions and classify them.

This is the acuity stream. It runs independently of the barcode stream so that
a tag with an unreadable barcode still yields an acuity, and a barcode with no
surrounding color still yields an ID.
"""

from __future__ import annotations

import cv2
import numpy as np

from dataclasses import dataclass

from .config import ColorBand, DetectorConfig
from .types import BBox, ColorRead


@dataclass
class Region:
    """A candidate tag found by color, before anything has been read from it."""

    bbox: BBox                 # axis-aligned, full-image coordinates
    color: ColorRead
    band_mask: np.ndarray      # this tag's colored pixels only, segmentation scale
    rrect: tuple               # minimum-area rect, so the crop can be de-rotated


def build_masks(
    image: np.ndarray, bands: tuple[ColorBand, ...]
) -> dict[str, np.ndarray]:
    """One binary mask per configured color band."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    masks: dict[str, np.ndarray] = {}
    for band in bands:
        acc = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in band.hsv_ranges:
            acc |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        masks[band.name] = acc
    return masks


def _clean(mask: np.ndarray, kernel_px: int) -> np.ndarray:
    """Drop speckle, then close only small gaps.

    Order matters and so does the kernel size. Opening first removes glare
    speckle and JPEG fringing. Closing then bridges the white dashes printed
    inside a tag's border -- but it must stay smaller than the gap *between*
    adjacent tags, or a column of stacked IMMEDIATE tags fuses into one region
    and every patient in it is lost.
    """
    k = max(3, kernel_px | 1)
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, close_k)
    return m


def find_tag_regions(image: np.ndarray, cfg: DetectorConfig) -> list[Region]:
    """Locate candidate tags.

    Returns (bbox, color_read, band_mask) per candidate, where `band_mask` holds
    only this tag's genuinely colored pixels -- NOT the filled outline. The
    distinction matters: downstream code locates the printed banner by finding
    rows that are almost entirely colored, and against a filled outline every
    row scores 100% and the "banner" degenerates to the whole tag.
    """
    h, w = image.shape[:2]
    frame_area = float(h * w)
    kernel_px = min(
        int(min(h, w) * cfg.close_kernel_frac), cfg.close_kernel_max_px
    )

    masks = build_masks(image, cfg.color_bands)
    band_by_name = {b.name: b for b in cfg.color_bands}

    candidates: list[Region] = []

    for name, raw in masks.items():
        band = band_by_name[name]
        cleaned = _clean(raw, kernel_px)

        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for raw_cnt in contours:
            # Work from the convex hull, not the raw contour. A tag's colored
            # field is a ring around a white data label, and the dashed border
            # printed through it can cut that ring -- leaving a C or L shaped
            # fragment whose fill is a small part of its bounding box. The hull
            # of that fragment still spans the tag, so shape gating stays
            # meaningful instead of throwing away real tags.
            cnt = cv2.convexHull(raw_cnt)
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = float(cw * ch)
            if not (
                cfg.min_tag_area_frac * frame_area
                <= area
                <= cfg.max_tag_area_frac * frame_area
            ):
                continue
            if ch == 0:
                continue
            aspect = cw / float(ch)
            if not (cfg.min_aspect <= aspect <= cfg.max_aspect):
                continue

            # A tag frame is a ring: its contour area is much smaller than its
            # bbox. Fill the contour so extent measures the *outline*, which is
            # what should be near-rectangular.
            filled = np.zeros((h, w), np.uint8)
            cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)
            filled_area = float(cv2.countNonZero(filled[y : y + ch, x : x + cw]))
            extent = filled_area / area
            if extent < cfg.min_extent:
                continue

            bbox = BBox(x, y, cw, ch)
            color = _score_region(image, masks, filled, bbox, band, cfg)
            if color is None:
                continue
            # This tag's own colored pixels: the band mask clipped to its outline,
            # so a neighbouring tag of the same color cannot contribute rows.
            band_mask = cv2.bitwise_and(cleaned, filled)
            candidates.append(
                Region(
                    bbox=bbox,
                    color=color,
                    band_mask=band_mask,
                    rrect=cv2.minAreaRect(cnt),
                )
            )

    return _suppress_overlaps(candidates)


def _score_region(
    image: np.ndarray,
    masks: dict[str, np.ndarray],
    filled: np.ndarray,
    bbox: BBox,
    band: ColorBand,
    cfg: DetectorConfig,
) -> ColorRead | None:
    """How strongly does this region belong to `band` versus the others?

    Scored over the *chromatic* pixels only -- the tag's white data field would
    otherwise dilute every color equally and make coverage meaningless.
    """
    sl = (slice(bbox.y, bbox.y + bbox.h), slice(bbox.x, bbox.x + bbox.w))
    region = filled[sl] > 0
    total = int(region.sum())
    if total == 0:
        return None

    scores: dict[str, int] = {}
    for name, m in masks.items():
        scores[name] = int(np.count_nonzero((m[sl] > 0) & region))

    chromatic = sum(scores.values())
    if chromatic == 0:
        return None

    own = scores[band.name]
    coverage = own / float(total)
    if coverage < band.min_coverage:
        return None

    # Margin over the runner-up: a clean single-color tag scores near 1.0, a
    # region straddling two tags of different colors scores near 0.
    runner_up = max((v for k, v in scores.items() if k != band.name), default=0)
    score = (own - runner_up) / float(chromatic)
    if score <= 0:
        return None

    return ColorRead(
        name=band.name,
        acuity_candidates=list(band.acuity_candidates),
        score=float(score),
        coverage=float(coverage),
    )


def _suppress_overlaps(
    candidates: list[Region], iou_thresh: float = 0.35
) -> list[Region]:
    """Greedy NMS. Several color bands can fire on one tag -- the red of an
    IMMEDIATE banner and the dark of its shadowed edge, say -- so keep the
    strongest claim on each area and drop the rest.
    """
    ordered = sorted(
        candidates,
        key=lambda r: (r.color.score * r.color.coverage, r.bbox.area),
        reverse=True,
    )
    kept: list[Region] = []
    for cand in ordered:
        if any(cand.bbox.iou(k.bbox) > iou_thresh for k in kept):
            continue
        # Also drop a candidate swallowed by one already kept.
        if any(
            k.bbox.contains_point(cand.bbox.cx, cand.bbox.cy)
            and cand.bbox.area < k.bbox.area * 0.85
            for k in kept
        ):
            continue
        kept.append(cand)
    return kept


def classify_crop(
    crop: np.ndarray, cfg: DetectorConfig
) -> ColorRead | None:
    """Classify an arbitrary crop (used for the barcode-only fallback path,
    where we sample the region around an orphan barcode).
    """
    if crop.size == 0:
        return None
    masks = build_masks(crop, cfg.color_bands)
    counts = {n: int(cv2.countNonZero(m)) for n, m in masks.items()}
    chromatic = sum(counts.values())
    if chromatic == 0:
        return None
    total = crop.shape[0] * crop.shape[1]

    best = max(counts, key=counts.get)
    band = next(b for b in cfg.color_bands if b.name == best)
    coverage = counts[best] / float(total)
    if coverage < band.min_coverage:
        return None
    runner_up = max((v for k, v in counts.items() if k != best), default=0)
    return ColorRead(
        name=best,
        acuity_candidates=list(band.acuity_candidates),
        score=float((counts[best] - runner_up) / float(chromatic)),
        coverage=float(coverage),
    )
