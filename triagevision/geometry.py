"""Orientation handling.

Tags photographed on the ground, on a patient, or on a triage tarp land at
arbitrary angles -- the sample photo has them at roughly 90 degrees, where an
OCR engine expecting horizontal text reads nothing at all. Every text read
therefore happens on a crop that has been warped upright first.

Two independent sources of angle, in order of preference:

  1. The barcode quad. The decoder reports the symbol's corners, and the bar
     direction is the tag's long axis. Precise, and available even when the
     tag's colored field is washed out.
  2. The color region's minimum-area rectangle. Available when the barcode did
     not decode, which is exactly when we most need the text.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

# (center_xy, size_wh, angle_degrees) -- OpenCV's RotatedRect tuple.
RRect = tuple[tuple[float, float], tuple[float, float], float]


def quad_angle(quad: list[list[int]]) -> float:
    """Angle of the symbol's reading direction: first corner to second.

    Decoders report a 1D symbol's corners in reading order, so this is a full
    360-degree orientation, not a 180-degree axis. That is what lets a tag be
    straightened without knowing anything about the frame, the other tags, or
    the field color.
    """
    (x0, y0), (x1, y1) = quad[0], quad[1]
    return math.degrees(math.atan2(y1 - y0, x1 - x0))


def rrect_from_quad(
    quad: list[list[int]],
    width_ratio: float = 1.45,
    aspect: float = 2.25,
    offset_frac: float = 0.17,
) -> RRect:
    """Grow a barcode quad into the tag it sits in, using layout priors.

    Everything is derived from the symbol's WIDTH. A 1D decoder localizes the
    bar direction precisely but its reported height is close to meaningless --
    measured across one sample sheet, the same tag design yielded symbol heights
    from 6px to 134px while widths held within 2%. Deriving the tag box from a
    height that varies twentyfold puts the banner outside the crop and the read
    fails for reasons that look like OCR trouble but are not.

    So: tag width = symbol width x `width_ratio`; tag height from `aspect`; and
    the tag centre sits `offset_frac` of its height off the symbol centre, on
    the banner side. Defaults are measured from standard START-scheme tags;
    override in DetectorConfig for a different vendor.

    The angle comes from the symbol's READING direction (first corner to
    second), not from a minimum-area rectangle. This is what makes the result
    fully rotation-invariant: a min-area rect is only defined up to 90 degrees,
    so it cannot say which way is up, and the layout prior that patched over
    that gap is unreliable once tags sit at arbitrary angles. The reading
    direction has no such ambiguity -- a tag rotated 180 degrees reads
    right-to-left, so aligning the bars left-to-right puts the banner on top by
    construction, for any rotation of the frame or of the individual tag.
    """  # noqa: D208
    pts = np.array(quad, dtype=np.float32)
    (x0, y0), (x1, y1) = quad[0], quad[1]
    angle = quad_angle(quad)
    w = float(math.hypot(x1 - x0, y1 - y0))

    cx = float(pts[:, 0].mean())
    cy = float(pts[:, 1].mean())

    new_w = w * width_ratio
    new_h = new_w / max(aspect, 0.1)

    # Step from the symbol centre toward the banner, perpendicular to the bars.
    rad = math.radians(angle)
    shift = offset_frac * new_h
    cx += math.sin(rad) * shift
    cy -= math.cos(rad) * shift
    return (cx, cy), (new_w, new_h), angle


def normalize_landscape(rrect: RRect) -> RRect:
    """Force the rect's first dimension to be its long side."""
    (cx, cy), (w, h), angle = rrect
    if w < h:
        return (cx, cy), (h, w), angle + 90.0
    return (cx, cy), (w, h), angle


def transform_point(m: np.ndarray, pt: tuple[float, float]) -> tuple[float, float]:
    """Push a point through a perspective matrix."""
    src = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
    out = cv2.perspectiveTransform(src, m)
    return float(out[0][0][0]), float(out[0][0][1])


def warp_upright(
    image: np.ndarray, rrect: RRect, pad_frac: float = 0.06, min_side: int = 24
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract a rotated region as an axis-aligned, landscape-oriented crop.

    Returns (crop, matrix); the matrix lets callers map points from the original
    frame into the crop -- which is how the tag's up/down orientation gets
    settled without spending OCR calls guessing. Returns None when the rect is
    degenerate or too small to be a tag.
    """
    (cx, cy), (w, h), angle = normalize_landscape(rrect)
    w = w * (1.0 + 2 * pad_frac)
    h = h * (1.0 + 2 * pad_frac)
    if w < min_side or h < min_side:
        return None

    out_w, out_h = int(round(w)), int(round(h))
    src = cv2.boxPoints(((cx, cy), (w, h), angle)).astype(np.float32)
    dst = np.array(
        [[0, out_h - 1], [0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1]],
        dtype=np.float32,
    )
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        image, m, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return (warped, m) if warped.size else None


def rotate_180(image: np.ndarray) -> np.ndarray:
    return cv2.rotate(image, cv2.ROTATE_180)


def scale_rrect(rrect: RRect, scale: float) -> RRect:
    """Map a rect measured on a downscaled image back to full resolution."""
    if scale == 1.0:
        return rrect
    inv = 1.0 / scale
    (cx, cy), (w, h), angle = rrect
    return (cx * inv, cy * inv), (w * inv, h * inv), angle


def rrect_to_bbox(rrect: RRect, shape: tuple[int, int]):
    """Axis-aligned bounding box of a rotated rect, clamped to the frame."""
    from .types import BBox

    h_img, w_img = shape
    pts = cv2.boxPoints(rrect)
    xs, ys = pts[:, 0], pts[:, 1]
    x0 = max(0, int(np.floor(xs.min())))
    y0 = max(0, int(np.floor(ys.min())))
    x1 = min(w_img, int(np.ceil(xs.max())))
    y1 = min(h_img, int(np.ceil(ys.max())))
    return BBox(x0, y0, max(0, x1 - x0), max(0, y1 - y0))


def suppress_glare(gray: np.ndarray) -> np.ndarray:
    """Flatten specular highlights before decoding or reading.

    A laminated or glossy tag under a scene light throws a blown-out band across
    the barcode -- in the sample photo that is exactly what costs us two of the
    nine symbols. Dividing by a heavily blurred copy removes the low-frequency
    illumination gradient while leaving the bar edges intact.

    The blur is computed on a thumbnail and scaled back up. It is a very
    low-frequency estimate by construction -- the sigma is a thirtieth of the
    frame -- so downsampling costs nothing in accuracy, while blurring 12MP
    directly with a kernel that wide takes over three seconds.
    """
    h, w = gray.shape[:2]
    small_w = max(32, w // 8)
    small_h = max(32, h // 8)
    small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
    blur_small = cv2.GaussianBlur(small, (0, 0), sigmaX=max(small_w, small_h) / 30.0)
    blur = cv2.resize(blur_small, (w, h), interpolation=cv2.INTER_LINEAR)

    flat = cv2.divide(gray, blur, scale=192)
    return cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
