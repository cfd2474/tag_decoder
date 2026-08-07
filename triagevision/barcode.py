"""Barcode decoding via zxing-cpp.

zxing-cpp is used rather than pyzbar because it needs no system library, it
returns symbol position (which we need for spatial matching to a tag), and it
handles rotated/perspective-skewed symbols natively.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

try:
    import zxingcpp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "zxing-cpp is required: pip install zxing-cpp"
    ) from exc

from . import geometry
from .types import BBox, BarcodeRead

# 1D linear symbologies typical of patient wristbands and triage tags, plus
# 2D codes some agencies print. Restricting the set speeds decoding and cuts
# false positives from stray line patterns.
DEFAULT_FORMATS = zxingcpp.BarcodeFormats(
    [
        zxingcpp.BarcodeFormat.Code128,
        zxingcpp.BarcodeFormat.Code39,  # what the sample tags actually carry
        zxingcpp.BarcodeFormat.Code93,
        zxingcpp.BarcodeFormat.ITF,
        zxingcpp.BarcodeFormat.Codabar,
        zxingcpp.BarcodeFormat.QRCode,
        zxingcpp.BarcodeFormat.DataMatrix,
        zxingcpp.BarcodeFormat.PDF417,
    ]
)


# Symbologies that carry a mandatory check character or error correction, so a
# clean decode is itself strong evidence the payload is right. Code 39, ITF and
# Codabar are absent on purpose: their check digits are OPTIONAL, and the tag
# stock here prints Code 39 without one, so a misread can decode cleanly to the
# wrong string. That is why the printed ID line is cross-checked.
SELF_CHECKING_FORMATS = frozenset(
    {
        "code128", "code93", "code32", "pzn",
        "qrcode", "microqrcode", "qrcodemodel1", "qrcodemodel2", "rmqrcode",
        "datamatrix", "pdf417", "micropdf417", "compactpdf417",
        "aztec", "azteccode",
        "ean13", "ean8", "upca", "upce", "isbn",
        "databar", "databarexp", "databarltd", "databaromni",
    }
)


def is_self_checking(format_name: str) -> bool:
    """True if a clean decode of this symbology implies a verified payload."""
    key = "".join(ch for ch in (format_name or "").lower() if ch.isalnum())
    return key in SELF_CHECKING_FORMATS


def _position_to_quad(pos) -> list[list[int]]:
    pts = []
    for name in ("top_left", "top_right", "bottom_right", "bottom_left"):
        p = getattr(pos, name)
        pts.append([int(p.x), int(p.y)])
    return pts


def _quad_to_bbox(quad: list[list[int]]) -> BBox:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    x, y = min(xs), min(ys)
    return BBox(x=x, y=y, w=max(xs) - x, h=max(ys) - y)


def _decode_pass(gray: np.ndarray, scale: float, formats) -> list[BarcodeRead]:
    if scale != 1.0:
        img = cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
    else:
        img = gray

    results = zxingcpp.read_barcodes(img, formats=formats)
    reads: list[BarcodeRead] = []
    for r in results:
        if not r.text:
            continue
        quad = _position_to_quad(r.position)
        if scale != 1.0:
            quad = [[int(px / scale), int(py / scale)] for px, py in quad]
        reads.append(
            BarcodeRead(
                text=r.text,
                format=str(r.format).split(".")[-1],
                bbox=_quad_to_bbox(quad),
                quad=quad,
            )
        )
    return reads


def _merge(found: list[BarcodeRead], new: list[BarcodeRead]) -> int:
    """Add reads that are not already present, returning how many were added.

    Identity is (text, overlapping position). Two tags legitimately carrying the
    same ID is a data-entry error worth surfacing, so we must not collapse by
    text alone -- but the same physical symbol re-found on a later pass must not
    be counted twice either. Overlap decides. (A coarse grid key does not: two
    reads of one symbol can straddle a cell boundary and survive as duplicates.)
    """
    added = 0
    for r in new:
        if any(
            e.text == r.text and (e.bbox.iou(r.bbox) > 0.20 or _close(e, r))
            for e in found
        ):
            continue
        found.append(r)
        added += 1
    return added


def _close(a: BarcodeRead, b: BarcodeRead) -> bool:
    """The same physical symbol, re-detected with different bounds by a later
    preprocessing pass.

    Tolerance comes from the symbol's LONG dimension. A 1D decoder pins the bar
    direction tightly but reports height erratically -- the same tag can come
    back 6px tall on one pass and 130px on the next -- so a height-derived
    tolerance lets one barcode survive as two entries and invents a patient.
    """
    tol = max(a.bbox.w, a.bbox.h, b.bbox.w, b.bbox.h, 1) * 0.15
    return abs(a.bbox.cx - b.bbox.cx) < tol and abs(a.bbox.cy - b.bbox.cy) < tol


def _rotate_pass(
    gray: np.ndarray, degrees: float, formats, scale: float = 1.0
) -> list[BarcodeRead]:
    """Decode a rotated copy, mapping any hits back to original coordinates.

    zxing scans along rows and internally retries at 90-degree steps, so it
    covers roughly +/-25 degrees around each cardinal direction and has blind
    bands in between. Tags on the ground sit at whatever angle they landed at:
    measured on the sample sheet, a frame rotated 37 degrees drops from nine
    decoded symbols to zero. Sweeping two intermediate angles closes the gaps.
    """
    # Sweeping at reduced scale. Rotating and re-scanning a full 12MP frame five
    # times dominates the whole pipeline, and a tag's symbol is ~700px wide at
    # native resolution -- half that is still ample for a 1D decoder. The sweep
    # only has to FIND the symbol; coordinates map back to full resolution and
    # everything downstream still works from the original pixels.
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    h, w = gray.shape[:2]
    centre = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(centre, degrees, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - centre[0]
    m[1, 2] += nh / 2 - centre[1]
    # Cubic, not linear: rotation resampling softens bar edges, and a 1D decoder
    # is measuring exactly those edge positions.
    rotated = cv2.warpAffine(
        gray, m, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=(128,)
    )

    reads = _decode_pass(rotated, 1.0, formats)
    if not reads:
        return []

    inverse = cv2.invertAffineTransform(m)
    inv_scale = 1.0 / scale
    for r in reads:
        pts = np.array([r.quad], dtype=np.float32)
        back = cv2.transform(pts, inverse)[0]
        r.quad = [[int(p[0] * inv_scale), int(p[1] * inv_scale)] for p in back]
        r.bbox = _quad_to_bbox(r.quad)
    return reads


# Two complementary tile scales, run in sequence and unioned. One grid is not
# enough and a "best" grid does not exist: measured on a fifteen-tag sheet,
# every single grid tried missed at least one symbol, and *which* one it missed
# changed with the grid, because zxing's response depends on how a symbol
# happens to sit inside its scan window. Coarse tiles catch symbols that need
# surrounding context; fine tiles catch the ones that only decode when cropped
# close. Together they found all fifteen; separately, never more than fourteen.
TILE_PLAN: tuple[tuple[tuple[int, int], float], ...] = (
    ((4, 4), 0.25),
    ((12, 7), 0.30),
)


def _tile_pass(
    gray: np.ndarray, formats, grid: tuple[int, int] = (3, 3), overlap: float = 0.25
) -> list[BarcodeRead]:
    """Decode over an overlapping tile grid, mapping hits back to frame coords.

    A whole-frame scan does not reliably return every symbol present. Measured
    on a fifteen-tag sheet, one pass returned two instances of some repeated
    payloads but only one of others -- and the skipped tags were invisible,
    because with colour off the barcode is the sole localizer. Both of the
    symbols it missed decoded immediately once cropped.

    Tiling fixes that on two fronts: each symbol occupies far more of the tile
    than it does the frame, and no symbol has to compete with fourteen others
    inside a single scan. Tiles overlap so a barcode straddling a seam is still
    whole in at least one of them.
    """
    h, w = gray.shape[:2]
    rows, cols = grid
    tile_h, tile_w = h / rows, w / cols
    pad_y, pad_x = tile_h * overlap, tile_w * overlap

    boxes = []
    for r in range(rows):
        for c in range(cols):
            y0 = max(0, int(r * tile_h - pad_y))
            y1 = min(h, int((r + 1) * tile_h + pad_y))
            x0 = max(0, int(c * tile_w - pad_x))
            x1 = min(w, int((c + 1) * tile_w + pad_x))
            if y1 - y0 >= 32 and x1 - x0 >= 32:
                boxes.append((x0, y0, x1, y1))

    def scan(box) -> list[BarcodeRead]:
        x0, y0, x1, y1 = box
        tile = gray[y0:y1, x0:x1]

        # Native, then glare-flattened. Illumination correction has to be
        # applied at TILE scale, not just to the whole frame: a specular
        # highlight is local, and flattening it across 12MP leaves the local
        # gradient intact. One symbol on the sample sheet decodes only under the
        # combination of a tight crop and per-tile glare suppression -- it fails
        # at native scale, and fails glare-corrected at full frame.
        reads = _decode_pass(tile, 1.0, formats)
        _merge(reads, _decode_pass(geometry.suppress_glare(tile), 1.0, formats))

        for read in reads:
            read.quad = [[px + x0, py + y0] for px, py in read.quad]
            read.bbox = _quad_to_bbox(read.quad)
        return reads

    # Decoding releases the GIL, so tiles genuinely run in parallel. Merging
    # stays on this thread: it is order-dependent and cheap.
    workers = min(len(boxes), (os.cpu_count() or 4))
    found: list[BarcodeRead] = []
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for reads in pool.map(scan, boxes):
                _merge(found, reads)
    else:
        for box in boxes:
            _merge(found, scan(box))
    return found


def decode_barcodes(
    image: np.ndarray,
    scales: tuple[float, ...] = (1.0, 2.0, 3.0),
    formats=DEFAULT_FORMATS,
    try_glare: bool = True,
    sweep_degrees: tuple[float, ...] = (15.0, 30.0, 45.0, 60.0, 75.0),
    tile_plan: tuple[tuple[tuple[int, int], float], ...] | None = TILE_PLAN,
) -> list[BarcodeRead]:
    """Decode every symbol in `image`, escalating effort until nothing new appears.

    Pass order is cheapest-first: native resolution, then glare-flattened, then
    upscaled. Each escalation only runs if the previous one is still finding new
    symbols or nothing has been found yet, so clean frames stay fast and only
    difficult ones pay for the extra passes.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    found: list[BarcodeRead] = []
    _merge(found, _decode_pass(gray, 1.0, formats))

    if try_glare:
        # Glossy laminated tags under a scene light blow out a band across the
        # symbol; flattening the illumination recovers those.
        _merge(found, _decode_pass(geometry.suppress_glare(gray), 1.0, formats))

    # A whole-frame scan silently skips symbols on a crowded sheet, so always
    # re-scan in tiles, at two scales. Like the rotation sweep, this has no
    # early exit: the frame gives no way to know how many tags it holds.
    for grid, overlap in tile_plan or ():
        _merge(found, _tile_pass(gray, formats, grid=grid, overlap=overlap))

    # Close the decoder's angular blind bands. The sweep always runs to
    # completion: there is no way to know how many tags a frame contains, so
    # "we already found some, stop looking" silently drops patients. Measured on
    # a five-tag frame, the cheap passes found two and an early exit stopped at
    # 30 degrees -- while the other three only decoded at 60 and 75. At half
    # scale the whole sweep costs about 0.6s, which is worth paying every time.
    sweep_scale = 0.5 if gray.shape[0] * gray.shape[1] > 3_000_000 else 1.0
    for degrees in sweep_degrees:
        _merge(found, _rotate_pass(gray, degrees, formats, scale=sweep_scale))

    # Upscaling a full-resolution phone frame is the most expensive thing here
    # (a 2x pass on 12MP is a 48MP scan), and it only pays off for symbols that
    # are small in pixel terms. Skip it when the frame is already large and the
    # cheap passes found something; a per-tag crop retry covers the stragglers.
    big_frame = gray.shape[0] * gray.shape[1] > 3_000_000
    if big_frame and found:
        return found

    for scale in scales:
        if scale == 1.0:
            continue
        added = _merge(found, _decode_pass(gray, scale, formats))
        if added == 0:
            break

    return found


def decode_in_roi(
    image: np.ndarray,
    bbox: BBox,
    scales: tuple[float, ...] = (1.0, 2.0, 4.0),
    formats=DEFAULT_FORMATS,
) -> list[BarcodeRead]:
    """Second-chance decode restricted to one tag crop, with coordinates mapped
    back to full-frame. Cropping raises the effective resolution the decoder
    sees, which recovers symbols the whole-frame pass missed.
    """
    h, w = image.shape[:2]
    x0, y0 = max(0, bbox.x), max(0, bbox.y)
    x1, y1 = min(w, bbox.x + bbox.w), min(h, bbox.y + bbox.h)
    if x1 <= x0 or y1 <= y0:
        return []

    crop = image[y0:y1, x0:x1]
    reads = decode_barcodes(crop, scales=scales, formats=formats)
    for r in reads:
        r.bbox = BBox(r.bbox.x + x0, r.bbox.y + y0, r.bbox.w, r.bbox.h)
        r.quad = [[px + x0, py + y0] for px, py in r.quad]
    return reads
