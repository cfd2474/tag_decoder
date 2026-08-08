"""Orchestration: locate tags, de-rotate them, read them, reconcile, score, emit.

Signal hierarchy, strongest to weakest:

  1. Banner text  -> acuity. Invariant to lighting. A closed five-word
                     vocabulary makes fuzzy matching safe, so a sloppy read
                     still lands on the right category.
  2. Barcode      -> patient ID. Open vocabulary, so it must be exact, and a
                     checksummed symbol either decodes or it does not.
  3. Printed ID   -> cross-check on the barcode, and a fallback when the symbol
                     is glare-blown or torn.
  4. Field color  -> corroborates acuity, and substitutes at reduced confidence
                     when the banner cannot be read.

Color never overrides a confident text read: a red tag under sodium light can
measure orange, and a green one under warm light can measure yellow, which is a
one-category error in the direction that gets someone hurt. Localization draws
on both color regions and barcode anchors so neither is a single point of
failure, and every crop is warped upright before any text is read.
"""

from __future__ import annotations

import difflib
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import barcode as barcode_mod
from . import color as color_mod
from . import geometry as geo
from . import layout
from . import preflight as preflight_mod
from . import textfind
from .config import DetectorConfig
from .ocr import TextReader, TextVerdict, best_verdict, get_reader
from .types import (
    Acuity,
    ImageQuality,
    BarcodeRead,
    BBox,
    ColorRead,
    DetectionResult,
    TagDetection,
)

log = logging.getLogger(__name__)

# Color segmentation gains nothing above this resolution and costs a lot. The
# barcode and OCR stages still run at native resolution, where detail matters.
MAX_SEGMENTATION_DIM = 1600

# Fractions of an upright tag's height. Layout is banner / printed ID / barcode.
BANNER_BAND = (0.00, 0.52)
ID_BAND = (0.26, 0.68)


def load_image(src) -> np.ndarray:
    """Accept a path, raw encoded bytes, or an existing BGR/gray/BGRA array."""
    if isinstance(src, np.ndarray):
        if src.ndim == 2:
            return cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        if src.ndim == 3 and src.shape[2] == 4:
            return cv2.cvtColor(src, cv2.COLOR_BGRA2BGR)
        return src
    if isinstance(src, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(bytes(src), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("could not decode image bytes")
        return img
    path = Path(src)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


@dataclass
class _Candidate:
    """A located tag, before anything has been read from it."""

    bbox: BBox
    rrect: tuple | None = None
    color: ColorRead | None = None
    barcode: BarcodeRead | None = None
    from_color: bool = False
    warp_matrix: np.ndarray | None = None   # frame -> upright crop, set on warp
    banner: "textfind.BannerHit | None" = None   # set when located by its word


@dataclass
class _Oriented:
    """An upright, banner-up tag crop and what settling its orientation told us."""

    crop: np.ndarray
    printed_id: str | None = None
    id_verified: bool = False       # printed ID confirmed against the payload
    stack: "layout.Stack | None" = None


class TriageTagDetector:
    """Stateless and thread-safe. Build once at process start, reuse per request.

    >>> det = TriageTagDetector()
    >>> det.detect("scene.jpg").roster()
    [{'patient_id': 'EA1568511', 'acuity': 'IMMEDIATE'}, ...]
    """

    def __init__(
        self,
        config: DetectorConfig | None = None,
        text_reader: TextReader | None = None,
    ) -> None:
        self.cfg = config or DetectorConfig()
        self.reader = text_reader or get_reader(self.cfg.ocr_backend)

    # ------------------------------------------------------------------ public

    def detect(self, image) -> DetectionResult:
        started = time.perf_counter()
        img = load_image(image)
        h, w = img.shape[:2]
        warnings: list[str] = []

        if not self.reader.available:
            warnings.append(
                "no OCR backend available -- acuity is falling back to field "
                "color, which is lighting-dependent; install tesseract"
            )

        # Readability check, started first and left to run while the barcode
        # stage works. It finishes in ~250ms against seconds for decoding, so a
        # good frame is never delayed -- only a doomed one is stopped, and by
        # then the expensive per-tag OCR has not begun.
        pre_future = None
        if self.cfg.preflight_enabled:
            pre_pool = ThreadPoolExecutor(max_workers=1)
            pre_future = pre_pool.submit(
                preflight_mod.check, img, self.cfg.text_keywords,
            )

        if self.cfg.use_color:
            seg_img, seg_scale = self._downscale_for_segmentation(img)
            regions = color_mod.find_tag_regions(seg_img, self.cfg)
        else:
            regions, seg_scale = [], 1.0
        barcodes = barcode_mod.decode_barcodes(img, scales=self.cfg.barcode_scales)

        pre = None
        if pre_future is not None:
            pre = pre_future.result()
            pre_pool.shutdown(wait=False)
            if pre.should_abort(self.cfg.preflight_abort_on):
                return DetectionResult(
                    tags=[], image_size=(w, h),
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                    warnings=[pre.advice] if pre.advice else [],
                    quality=ImageQuality(
                        rating="poor" if pre.rating == "degraded" else "empty",
                        barcode_decode_rate=0.0, id_verified_rate=0.0,
                        tags_found=0, sharpness=pre.sharpness,
                        retake_recommended=True, advice=pre.advice,
                    ),
                    aborted=True, preflight=pre,
                )

        candidates = self._localize(img, seg_scale, regions, barcodes)

        # Second localizer: find tags by their printed banner word. A soft photo
        # can leave most barcodes undecodable while every banner stays legible,
        # and with the barcode as sole localizer those patients vanish silently.
        if self.cfg.use_text_localizer and self.reader.available:
            candidates.extend(self._localize_by_text(img, candidates))
        # Any barcode that decoded reveals this batch's ID shape, which is what
        # lets OCR-recovered IDs be checked strictly. See textfind.shape_template.
        self._id_template = textfind.shape_template([b.text for b in barcodes])

        tags = [t for t in self._read_all(img, candidates) if t]
        tags = [t for t in tags if t.confidence >= self.cfg.min_confidence]
        tags = self._merge_same_tag(tags)
        tags.sort(key=lambda t: (t.bbox.cy, t.bbox.cx))

        quality = self._assess_quality(img, tags)
        if quality.retake_recommended and quality.advice:
            warnings.append(quality.advice)

        return DetectionResult(
            tags=tags,
            image_size=(w, h),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            warnings=warnings,
            quality=quality,
            preflight=pre,
        )

    # ------------------------------------------------------------ text anchors

    def _localize_by_text(
        self, img: np.ndarray, existing: list[_Candidate]
    ) -> list[_Candidate]:
        """Add a candidate for every banner word not already covered by a tag."""
        def uncovered(hs):
            return [
                h for h in hs
                if not any(
                    c.bbox.contains_point(h.bbox.cx, h.bbox.cy, pad=h.bbox.h * 0.5)
                    for c in existing
                )
            ]

        # Cheap probe first: one OCR call. On a sharp frame where the barcodes
        # already found every tag it finds nothing new and we stop, which keeps
        # the common case fast. Escalating only on EVIDENCE of a missed tag is
        # different from "we found some, stop looking" -- the probe has to show
        # a banner sitting outside every located tag before we spend more.
        probe = textfind.find_banners(
            img, self.cfg.text_keywords, scan_width=self.cfg.text_probe_width,
            psms=self.cfg.text_scan_psms, workers=self.cfg.max_workers,
            probe_only=True,
        )
        hits = probe
        if uncovered(probe):
            hits = textfind.find_banners(
                img, self.cfg.text_keywords, scan_width=self.cfg.text_scan_width,
                psms=self.cfg.text_scan_psms, workers=self.cfg.max_workers,
            )
        if not hits:
            return []

        extra: list[_Candidate] = []
        for hit in hits:
            # The banner sits inside the tag, so a hit whose centre falls in an
            # already-located tag is that tag, not a new one.
            if any(
                c.bbox.contains_point(hit.bbox.cx, hit.bbox.cy, pad=hit.bbox.h * 0.5)
                for c in existing
            ):
                continue
            if any(
                c.banner is not None and c.bbox.iou(hit.bbox) > 0.2 for c in extra
            ):
                continue
            extra.append(
                _Candidate(bbox=self._tag_box_from_banner(hit, img.shape[:2]), banner=hit)
            )
        return extra

    @staticmethod
    def _tag_box_from_banner(hit, shape: tuple[int, int]) -> BBox:
        """Approximate tag bounds from its banner word.

        The banner spans most of the tag's width and sits at the top, so the tag
        extends downward roughly three word-heights to cover the ID line and the
        barcode beneath it.
        """
        h_img, w_img = shape
        b = hit.bbox
        x0 = max(0, int(b.x - b.w * 0.22))
        y0 = max(0, int(b.y - b.h * 0.70))
        x1 = min(w_img, int(b.x + b.w * 1.22))
        y1 = min(h_img, int(b.y + b.h * 3.10))
        return BBox(x0, y0, max(1, x1 - x0), max(1, y1 - y0))

    # ------------------------------------------------------------------ quality

    def _assess_quality(self, img: np.ndarray, tags: list[TagDetection]):
        """Rate how readable this frame was, so a caller can ask for a retake."""
        n = len(tags)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if n == 0:
            return ImageQuality(
                rating="empty", barcode_decode_rate=0.0, id_verified_rate=0.0,
                tags_found=0, sharpness=sharp, retake_recommended=True,
                advice="no triage tags were found in this image; retake with the "
                       "tags filling more of the frame",
            )

        decoded = sum(1 for t in tags if t.barcode is not None)
        verified = sum(
            1 for t in tags
            if not any("unverified" in w or "verify this tag" in w for w in t.warnings)
            and t.patient_id
        )
        bc_rate, id_rate = decoded / n, verified / n

        if bc_rate >= 0.90:
            rating, retake, advice = "good", False, None
        elif bc_rate >= 0.60:
            rating, retake = "marginal", True
            advice = (
                f"image quality marginal: only {decoded} of {n} tags had a "
                "readable barcode, so some patient IDs come from OCR and may "
                "contain character errors; a sharper, closer photo is advised"
            )
        else:
            rating, retake = "poor", True
            advice = (
                f"image quality poor: only {decoded} of {n} tags had a readable "
                "barcode. Patient IDs are largely from OCR and tags may be "
                "missing entirely. Retake: hold steady, fill the frame with the "
                "tags, and avoid glare across the barcodes"
            )
        return ImageQuality(
            rating=rating, barcode_decode_rate=bc_rate, id_verified_rate=id_rate,
            tags_found=n, sharpness=sharp, retake_recommended=retake, advice=advice,
        )

    def detect_many(self, images) -> list[DetectionResult]:
        return [self.detect(i) for i in images]

    # ------------------------------------------------------------- localization

    def _localize(
        self, img: np.ndarray, seg_scale: float, regions, barcodes
    ) -> list[_Candidate]:
        """Pair color regions with barcodes, then keep the leftovers of both.

        Two independent localizers, so neither is a single point of failure: a
        washed-out tag still has a barcode, and a glare-blown barcode still has
        a colored field.
        """
        candidates: list[_Candidate] = []
        used: set[int] = set()

        for region in regions:
            bbox = self._rescale_bbox(region.bbox, seg_scale)
            rrect = geo.scale_rrect(region.rrect, seg_scale)

            inside = self._barcodes_inside(bbox, barcodes, used)

            if len(inside) > 1:
                # Adjacent same-color tags fuse into one color blob -- three
                # stacked IMMEDIATEs read as one region and two patients vanish.
                # The barcodes split them reliably: a merged blob is by
                # construction all one color, so each piece safely inherits the
                # region's acuity while taking its own geometry from its symbol.
                for i in inside:
                    used.add(i)
                    sub = self._tag_rrect(barcodes[i])
                    bc = barcodes[i]
                    candidates.append(
                        _Candidate(
                            bbox=geo.rrect_to_bbox(sub, img.shape[:2]),
                            rrect=sub,
                            color=region.color,
                            barcode=bc,
                            from_color=True,
                        )
                    )
                continue

            if inside:
                used.add(inside[0])
                bc = barcodes[inside[0]]
            else:
                # Second chance inside this crop alone, where the symbol fills
                # far more of the frame than it does full-scene.
                bc = self._first_valid(
                    barcode_mod.decode_in_roi(img, self._pad(bbox, img, 0.04))
                )
            candidates.append(
                _Candidate(
                    bbox=bbox, rrect=rrect, color=region.color, barcode=bc,
                    from_color=True,
                )
            )

        if self.cfg.emit_orphan_barcodes:
            for i, bc in enumerate(barcodes):
                if i in used or not self.cfg.is_valid_patient_id(bc.text):
                    continue
                rrect = self._tag_rrect(bc)
                candidates.append(
                    _Candidate(
                        bbox=geo.rrect_to_bbox(rrect, img.shape[:2]),
                        rrect=rrect,
                        barcode=bc,
                    )
                )

        return candidates

    # ---------------------------------------------------------------- reading

    def _read_all(
        self, img: np.ndarray, candidates: list[_Candidate]
    ) -> list[TagDetection | None]:
        """Read every candidate, in parallel when it is worth the overhead.

        Reading a tag is dominated by OCR, and each OCR call spawns a separate
        tesseract process, so the work is genuinely parallel -- the GIL is not
        held across it. A frame with a dozen casualties on it is exactly when
        latency matters most.
        """
        workers = self.cfg.max_workers
        if workers == 1 or len(candidates) < 2:
            return [self._read(img, c) for c in candidates]

        if workers <= 0:
            workers = min(len(candidates), (os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda c: self._read(img, c), candidates))

    def _read(self, img: np.ndarray, cand: _Candidate) -> TagDetection | None:
        if cand.banner is not None and cand.barcode is None:
            return self._read_text_anchored(img, cand)

        warns: list[str] = []

        oriented = self._resolve_stack(img, cand)
        verdict = self._read_banner(oriented.crop, oriented.stack)

        patient_id, id_warns, cross = self._resolve_id(cand, oriented)
        warns.extend(id_warns)
        if patient_id is None and not self.cfg.emit_unidentified_tags:
            return None
        if self.cfg.use_color and not cand.from_color:
            warns.append("barcode not enclosed by a recognized tag region")

        acuity, acuity_warns, corroboration = self._decide_acuity(verdict, cand.color)
        warns.extend(acuity_warns)

        return TagDetection(
            patient_id=patient_id,
            acuity=acuity,
            confidence=self._score(
                has_id=patient_id is not None,
                verdict=verdict,
                corroboration=corroboration,
                cross_check=cross,
                color=cand.color,
                located_by_color=cand.from_color or not self.cfg.use_color,
                color_enabled=self.cfg.use_color,
            ),
            bbox=cand.bbox,
            color=cand.color,
            barcode=cand.barcode,
            banner_text=verdict.text or None,
            warnings=warns,
            id_source=("barcode" if (patient_id and cand.barcode) else
                       ("ocr" if patient_id else None)),
            found_by="barcode" if cand.barcode else "color",
        )

    def _upright_crop(self, img: np.ndarray, cand: _Candidate) -> np.ndarray:
        """De-rotate the tag into a landscape crop.

        Without this, a tag lying at 90 degrees -- most of them, in a photo
        taken looking down at a tarp -- reads as nothing at all: an OCR engine
        expecting horizontal text sees vertical strokes. The angle comes from
        the barcode's own quad, so each tag is straightened against itself and
        neighbours at different angles do not interfere.

        When a barcode is present the crop is taken SYMMETRIC about it, tall
        enough to contain the tag whichever way it faces. Deciding up/down here
        from a prior would be circular -- the prior is what built the box -- so
        that call is deferred to the ink profile, which reads it off the tag.
        """
        rrect = cand.rrect
        if cand.barcode is not None:
            rrect = self._tag_rrect(cand.barcode)

        if rrect is not None:
            warped = geo.warp_upright(img, rrect)
            if warped is not None and warped[0].size:
                crop, m = warped
                cand.warp_matrix = m
                return crop

        b = self._clamp(cand.bbox, *img.shape[:2])
        crop = img[b.y : b.y + b.h, b.x : b.x + b.w]
        if crop.size and crop.shape[0] > crop.shape[1]:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        return crop

    def _read_text_anchored(
        self, img: np.ndarray, cand: _Candidate
    ) -> TagDetection | None:
        """Read a tag that only its banner word located.

        The acuity is already known -- the word is what found the tag. What is
        missing is the ID, which comes from the printed line directly beneath.
        That ID is OCR off an open-vocabulary serial, exactly the case the
        barcode normally protects against, so it is always marked and flagged.
        """
        hit = cand.banner
        warns = [
            "tag located by its printed banner, not a barcode -- the symbol "
            "did not decode"
        ]

        band = textfind.id_band(hit, img.shape[:2])
        template = getattr(self, "_id_template", None)
        patient_id = textfind.read_id_text(
            img, band, self.cfg.is_valid_patient_id, template=template
        )

        if patient_id:
            warns.append(
                f"patient id {patient_id!r} was read by OCR from the printed "
                "line, with no barcode to confirm it; it may contain character "
                "errors (0/O, 1/I, 5/S, 8/B) -- verify before use"
            )
        else:
            warns.append(
                "no barcode, and the printed id could not be read to a form "
                "matching the other tags in this frame; patient id unknown"
                if getattr(self, "_id_template", None)
                else "no barcode and no readable printed id on this tag"
            )
            if not self.cfg.emit_unidentified_tags:
                return None

        # Capped well below a barcode-backed read: the acuity is solid (the word
        # was matched with a clear margin) but the identity is not.
        confidence = 0.55 if patient_id else 0.30
        if hit.margin < 0.30:
            confidence -= 0.05

        return TagDetection(
            patient_id=patient_id,
            acuity=hit.acuity,
            confidence=max(0.0, min(1.0, confidence)),
            bbox=cand.bbox,
            color=None,
            barcode=None,
            banner_text=hit.text,
            warnings=warns,
            id_source="ocr" if patient_id else None,
            found_by="text",
        )

    def _resolve_stack(
        self, img: np.ndarray, cand: _Candidate
    ) -> tuple[np.ndarray, layout.Stack | None]:
        """Produce an upright, banner-up crop of the tag, plus its layout.

        A tag's three elements -- banner, printed ID, barcode -- are stacked in
        a fixed order and rotate together, so orientation is a property of the
        tag alone. That is what makes a photo with independently rotated tags
        tractable: each is straightened against its own contents rather than
        against the frame.

        The crop is taken symmetric about the barcode so it contains the tag
        whichever way it faces, and the ink profile then says which side holds
        the text. Where a barcode decoded, the choice is *checked* rather than
        assumed -- the printed ID must match the payload we already know.
        """
        crop = self._upright_crop(img, cand)
        if crop is None or not crop.size:
            return _Oriented(crop=crop)

        if cand.barcode is not None:
            # The box was built from the barcode's reading direction, so the
            # crop is already banner-up at any rotation -- no ambiguity left.
            return _Oriented(crop=crop)

        # No decodable symbol, so no reading direction, and a color region's
        # rectangle is only defined up to 180 degrees. Here the ink profile is
        # the only thing that can say which end is which: the banner and ID sit
        # together at one end of the tag, and the barcode's blank white field at
        # the other, so the text-heavy end is the top.
        stack = layout.resolve_stack(crop, crop.shape[0] * 0.72)
        if stack.flipped:
            h = crop.shape[0]
            crop = geo.rotate_180(crop)
            stack = layout.Stack(
                banner=layout.flip_row(stack.banner, h) if stack.banner else None,
                ident=layout.flip_row(stack.ident, h) if stack.ident else None,
                flipped=False,
                rows=stack.rows,
            )
        return _Oriented(crop=crop, stack=stack if stack.banner else None)

    def _read_banner(
        self, oriented: np.ndarray, stack: layout.Stack | None
    ) -> TextVerdict:
        """Read the acuity word from the banner row."""
        blank = TextVerdict("", None, 0.0)
        if (
            self.cfg.ocr_policy == "never"
            or not self.reader.available
            or oriented is None
            or not oriented.size
        ):
            return blank

        # Fixed band first, ink-profile row second. The band is derived from
        # measured tag proportions and is reliably right; the row detector is
        # more principled but empirically less accurate, so it earns a place as
        # a fallback for odd layouts rather than as the primary crop.
        h = oriented.shape[0]
        crops = [self._band(oriented, *BANNER_BAND)]
        if stack and stack.banner:
            y0, y1 = layout.pad_row(stack.banner, h)
            if y1 - y0 > 8:
                crops.append(oriented[y0:y1])

        best = blank
        for crop in crops:
            v = best_verdict(self.reader, [crop], self.cfg.text_keywords, max_calls=4)
            if v.exact:
                return v
            if v.score > best.score:
                best = v

        if best.acuity is None:
            # Last resort: the tag may be upside down because the ink profile
            # picked the wrong side. Re-check the opposite end.
            flipped = geo.rotate_180(oriented)
            v = best_verdict(
                self.reader,
                [self._band(flipped, *BANNER_BAND)],
                self.cfg.text_keywords,
                max_calls=4,
            )
            if v.score > best.score:
                best = v
        return best

    def _resolve_id(
        self, cand: _Candidate, oriented: _Oriented
    ) -> tuple[str | None, list[str], str]:
        """Decide the patient ID from the barcode and the printed ID line.

        The printed ID is the redundancy that catches a silent barcode misread.
        Code 39 without a check digit -- which is what the sample tags carry --
        can decode cleanly to the wrong string, and a wrong patient ID is worse
        than a missing one. Returns (id, warnings, cross_check) where
        cross_check is "agree" | "disagree" | "none" | "ocr_only".
        """
        warns: list[str] = []
        bc_text: str | None = None
        if cand.barcode:
            if self.cfg.is_valid_patient_id(cand.barcode.text):
                bc_text = cand.barcode.text.strip()
            else:
                warns.append(
                    f"barcode payload failed id pattern: {cand.barcode.text!r}"
                )

        # Orientation resolution already read this line and, when it matched,
        # confirmed it against the payload -- no need to OCR it a second time.
        if oriented.id_verified and bc_text:
            return bc_text, warns, "agree"
        printed = oriented.printed_id or self._read_printed_id(
            oriented.crop, bc_text, oriented.stack
        )

        if bc_text and printed:
            # Confirmation requires an EXACT match, not a high similarity.
            # Patient IDs are sequential vendor serials, so the single most
            # likely barcode misread -- one digit off -- yields a near-identical
            # string: EA1568513 against EA1568512 scores 0.89. A similarity
            # threshold loose enough to absorb OCR noise would therefore bless
            # exactly the error this check exists to catch.
            a = self._normalize_id(bc_text)
            b = self._normalize_id(printed)
            if a == b:
                return bc_text, warns, "agree"

            ratio = difflib.SequenceMatcher(None, a, b).ratio()
            if ratio < 0.50:
                # Two genuinely different strings. Could be a barcode misread or
                # the ID line of a neighbouring tag caught in the crop; either
                # way a human needs to look.
                warns.append(
                    f"barcode says {bc_text!r} but the printed id reads "
                    f"{printed!r} (similarity {ratio:.2f}); barcode kept -- "
                    "verify this tag"
                )
                return bc_text, warns, "disagree"

            # Close but not identical: almost always OCR noise on the printed
            # line rather than a bad decode, so this is not evidence against the
            # barcode -- but it is not confirmation either, and saying so beats
            # a false alarm that teaches operators to ignore the warnings.
            self._warn_unverified(cand, warns)
            return bc_text, warns, "none"

        if bc_text:
            self._warn_unverified(cand, warns)
            return bc_text, warns, "none"

        if printed:
            warns.append(
                f"no decodable barcode; patient id {printed!r} came from OCR of "
                "the printed line and may contain character errors"
            )
            return printed, warns, "ocr_only"

        warns.append("no readable barcode and no readable printed id")
        return None, warns, "none"

    @staticmethod
    def _normalize_id(text: str) -> str:
        """Case- and punctuation-insensitive form, for comparison only."""
        return "".join(ch for ch in text.upper() if ch.isalnum())

    @staticmethod
    def _warn_unverified(cand: _Candidate, warns: list[str]) -> None:
        """Flag a patient ID that nothing independently corroborated.

        Patient IDs are vendor pre-printed serials in an unknown format, so the
        ID pattern must stay permissive and cannot reject a bad decode on shape.
        With a symbology that has no mandatory check character -- Code 39, which
        is what this stock uses -- that leaves the printed ID line as the only
        confirmation. When it could not be read, the ID rests on a single
        unverified decode, and the operator should be told rather than left to
        assume it was cross-checked.
        """
        if cand.barcode is None:
            return
        if barcode_mod.is_self_checking(cand.barcode.format):
            return
        warns.append(
            f"patient id came from a single {cand.barcode.format} decode with no "
            "check character, and the printed id line could not be read to "
            "confirm it; treat the id as unverified"
        )

    def _read_printed_id(
        self,
        oriented: np.ndarray,
        bc_text: str | None,
        stack: layout.Stack | None = None,
    ) -> str | None:
        """OCR the human-readable ID line that sits between banner and barcode.

        Uses the ink-profile row when one was found, since vendors place the
        line at different heights, and falls back to a fixed band otherwise.
        """
        read_id = getattr(self.reader, "read_id", None)
        if read_id is None or oriented is None or not oriented.size:
            return None
        if self.cfg.ocr_policy == "never":
            return None

        band = self._band(oriented, *ID_BAND)
        if band is None or band.size == 0:
            if not (stack and stack.ident):
                return None
            y0, y1 = layout.pad_row(stack.ident, oriented.shape[0])
            band = oriented[y0:y1]
        if band is None or not band.size:
            return None

        candidates = [
            t
            for t in read_id(band)
            if self.cfg.is_valid_patient_id(t) and not self._looks_like_banner(t)
        ]
        if not candidates:
            return None
        if bc_text:
            # Pick the token the barcode most plausibly corresponds to, so a
            # stray fragment cannot masquerade as a mismatch.
            return max(
                candidates,
                key=lambda t: difflib.SequenceMatcher(
                    None, t.upper(), bc_text.upper()
                ).ratio(),
            )
        return max(candidates, key=len)

    def _looks_like_banner(self, token: str) -> bool:
        """Reject acuity words that bled into the ID band from the banner above."""
        up = token.upper()
        return any(
            difflib.SequenceMatcher(None, up, word).ratio() >= 0.75
            for word in self.cfg.text_keywords
        )

    @staticmethod
    def _band(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
        h = img.shape[0]
        y0, y1 = int(h * lo), max(int(h * hi), int(h * lo) + 1)
        return img[y0:y1]

    # ------------------------------------------------------------- reconciling

    def _decide_acuity(
        self, verdict: TextVerdict, color: ColorRead | None
    ) -> tuple[Acuity, list[str], str]:
        """Text decides. Color corroborates, disputes, or substitutes.

        Returns (acuity, warnings, corroboration) where corroboration is
        "agree" | "disagree" | "none", consumed by the confidence model.
        """
        warns: list[str] = []
        options = color.acuity_candidates if color else []

        if verdict.acuity is not None:
            if not options:
                if not verdict.confident:
                    # An ambiguous match with nothing to corroborate it is a
                    # guess. On a triage tag, a confident wrong category is
                    # worse than admitting we could not read it. "Ambiguous"
                    # means close to more than one word, not merely a low score.
                    warns.append(
                        f"banner text {verdict.text!r} is ambiguous: it matched "
                        f"{verdict.acuity.value} at {verdict.score:.2f} but sits "
                        f"only {verdict.margin:.2f} ahead of the next category, "
                        "and no field color corroborates it; reporting UNKNOWN"
                    )
                    return Acuity.UNKNOWN, warns, "none"
                return verdict.acuity, warns, "none"
            if verdict.acuity in options:
                return verdict.acuity, warns, "agree"
            warns.append(
                f"banner text reads {verdict.acuity.value} but field color "
                f"'{color.name}' implies {[c.value for c in options]}; text kept "
                "-- color is lighting-dependent"
            )
            if not verdict.confident:
                warns.append(
                    f"text match was weak ({verdict.score:.2f}); manual review advised"
                )
            return verdict.acuity, warns, "disagree"

        if verdict.text:
            warns.append(f"banner text {verdict.text!r} matched no known category")
        else:
            warns.append("banner text unreadable")

        if self.cfg.require_text:
            warns.append("require_text is set; refusing to infer acuity from color")
            return Acuity.UNKNOWN, warns, "none"

        if len(options) == 1:
            warns.append(
                f"acuity inferred from field color '{color.name}' alone -- "
                "verify before acting"
            )
            return options[0], warns, "none"

        if len(options) > 1:
            warns.append(
                f"field color '{color.name}' is ambiguous between "
                f"{[c.value for c in options]} and the text is unreadable; "
                "manual review required"
            )
            return Acuity.UNKNOWN, warns, "none"

        warns.append("no text and no field color; acuity undetermined")
        return Acuity.UNKNOWN, warns, "none"

    # ---------------------------------------------------------------- scoring

    @staticmethod
    def _score(
        *,
        has_id: bool,
        verdict: TextVerdict,
        corroboration: str,
        cross_check: str,
        color: ColorRead | None,
        located_by_color: bool,
        color_enabled: bool = True,
    ) -> float:
        """Interpretable confidence, weighted toward the invariant signals.

        With color enabled:
          0.35  patient ID recovered at all
          0.50  acuity from banner text (full on an exact word match, scaled by
                similarity on a fuzzy one, capped low when only color spoke)
          0.15  text/color agreement on acuity

        With color disabled the 0.15 corroboration weight has nowhere to go, so
        it moves onto the two signals that remain (0.40 / 0.55) -- otherwise
        every tag would be capped at 0.90 and the review threshold would lose
        its meaning.

        Either way:
         +0.05  barcode agrees with the printed ID
         -0.20  barcode disagrees with the printed ID

        With color on, a tag read from color alone tops out near 0.57 by design:
        below the 0.60 line where an operator should look at the frame, which is
        the honest position when the only evidence is lighting-dependent.
        """
        w_id, w_text = (0.35, 0.50) if color_enabled else (0.40, 0.55)
        score = w_id if has_id else 0.0

        if verdict.acuity is not None:
            quality = 1.0 if verdict.exact else max(0.0, min(1.0, verdict.score))
            score += w_text * (quality if verdict.confident else 0.65 * quality)
        elif color and len(color.acuity_candidates) == 1:
            reliability = min(1.0, color.score) * min(1.0, color.coverage / 0.35)
            score += w_text * 0.45 * reliability

        if corroboration == "agree":
            score += 0.15
        elif corroboration == "disagree":
            score -= 0.15

        if cross_check == "agree":
            score += 0.05
        elif cross_check == "disagree":
            score -= 0.20
        elif cross_check == "ocr_only":
            score -= 0.10

        if not located_by_color:
            score -= 0.05

        return max(0.0, min(1.0, score))

    # ---------------------------------------------------------------- geometry

    @staticmethod
    def _downscale_for_segmentation(img: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest <= MAX_SEGMENTATION_DIM:
            return img, 1.0
        scale = MAX_SEGMENTATION_DIM / longest
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return small, scale

    @staticmethod
    def _rescale_bbox(b: BBox, scale: float) -> BBox:
        if scale == 1.0:
            return b
        inv = 1.0 / scale
        return BBox(int(b.x * inv), int(b.y * inv), int(b.w * inv), int(b.h * inv))

    @staticmethod
    def _clamp(b: BBox, h: int, w: int) -> BBox:
        x = max(0, min(b.x, w - 1))
        y = max(0, min(b.y, h - 1))
        return BBox(x, y, max(1, min(b.w, w - x)), max(1, min(b.h, h - y)))

    def _tag_rrect(self, bc: BarcodeRead):
        """Infer the tag's oriented box from its barcode, using layout priors."""
        return geo.rrect_from_quad(
            bc.quad,
            width_ratio=self.cfg.tag_width_per_barcode_width,
            aspect=self.cfg.tag_aspect_prior,
            offset_frac=self.cfg.barcode_offset_frac,
        )

    def _barcodes_inside(
        self, bbox: BBox, barcodes: list[BarcodeRead], used: set[int]
    ) -> list[int]:
        """Indices of unclaimed barcodes whose centre falls in this region,
        nearest to the region centre first.
        """
        hits: list[tuple[float, int]] = []
        for i, bc in enumerate(barcodes):
            if i in used:
                continue
            pad = bc.bbox.h * self.cfg.match_pad_frac
            if not bbox.contains_point(bc.bbox.cx, bc.bbox.cy, pad=pad):
                continue
            d = (bc.bbox.cx - bbox.cx) ** 2 + (bc.bbox.cy - bbox.cy) ** 2
            hits.append((d, i))
        return [i for _, i in sorted(hits)]

    def _first_valid(self, reads: list[BarcodeRead]) -> BarcodeRead | None:
        for r in reads:
            if self.cfg.is_valid_patient_id(r.text):
                return r
        return reads[0] if reads else None

    @staticmethod
    def _pad(b: BBox, img: np.ndarray, frac: float) -> BBox:
        h, w = img.shape[:2]
        px, py = int(b.w * frac), int(b.h * frac)
        x, y = max(0, b.x - px), max(0, b.y - py)
        return BBox(x, y, min(w - x, b.w + 2 * px), min(h - y, b.h + 2 * py))

    @staticmethod
    def _merge_same_tag(tags: list[TagDetection]) -> list[TagDetection]:
        """Collapse repeat detections of ONE physical tag, keeping the best read.

        A single tag can reach this point twice -- once from its color region
        and once from a second-chance decode inside a neighbour's crop, say.
        Same ID *and* overlapping geometry means one piece of card seen twice,
        so the weaker read is dropped; otherwise it would be counted as two
        patients.

        This is deliberately NOT duplicate-ID handling. Two tags carrying the
        same patient ID at different places in the frame are both reported, as
        separate line items, whether or not their acuities agree -- reconciling
        that is another system's job, and this one's contract is to report what
        it saw.
        """
        kept: list[TagDetection] = []
        for tag in sorted(tags, key=lambda t: t.confidence, reverse=True):
            twin = next(
                (
                    k
                    for k in kept
                    if k.patient_id
                    and k.patient_id == tag.patient_id
                    and k.bbox.iou(tag.bbox) > 0.25
                ),
                None,
            )
            if twin is None:
                kept.append(tag)
        return kept

def annotate(image, result: DetectionResult) -> np.ndarray:
    """Draw detections, for debugging or an operator-facing preview."""
    img = load_image(image).copy()
    scale = max(img.shape[:2]) / 1200.0
    thick = max(2, int(3 * scale))
    font = max(0.6, 0.7 * scale)
    palette = {
        Acuity.IMMEDIATE: (0, 0, 220),
        Acuity.DELAYED: (0, 210, 235),
        Acuity.MINOR: (0, 170, 0),
        Acuity.EXPECTANT: (150, 150, 150),   # grey field
        Acuity.DEAD: (20, 20, 20),           # black field, one protocol...
        Acuity.MORGUE: (55, 55, 55),         # ...and the other. Kept distinct.
        Acuity.UNKNOWN: (200, 0, 200),
    }
    for t in result.tags:
        bgr = palette.get(t.acuity, (255, 255, 255))
        b = t.bbox
        cv2.rectangle(img, (b.x, b.y), (b.x + b.w, b.y + b.h), bgr, thick)
        label = f"{t.patient_id or '???'} {t.acuity.value} {t.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font, thick)
        ly = max(th + 8, b.y)
        cv2.rectangle(img, (b.x, ly - th - 8), (b.x + tw + 10, ly + 6), bgr, -1)
        cv2.putText(img, label, (b.x + 5, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    font, (255, 255, 255), thick, cv2.LINE_AA)
    return img
