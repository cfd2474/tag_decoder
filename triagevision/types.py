"""Core data types for triage tag detection."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, asdict
from typing import Any


class Acuity(str, enum.Enum):
    """Triage categories, exactly as printed on the tag.

    DEAD and MORGUE denote the same clinical outcome but come from two different
    triage protocols, and they are deliberately NOT merged: this output feeds
    another system, and collapsing them would destroy the information about
    which protocol produced the tag. Anything that needs to treat them alike
    should do so explicitly -- see `is_deceased`.
    """

    IMMEDIATE = "IMMEDIATE"
    DELAYED = "DELAYED"
    MINOR = "MINOR"
    EXPECTANT = "EXPECTANT"
    DEAD = "DEAD"
    MORGUE = "MORGUE"
    UNKNOWN = "UNKNOWN"

    @property
    def is_deceased(self) -> bool:
        """True for both protocols' deceased category, for callers that need it."""
        return self in (Acuity.DEAD, Acuity.MORGUE)


@dataclass(frozen=True)
class BBox:
    """Axis-aligned box in pixel coordinates."""

    x: int
    y: int
    w: int
    h: int

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> int:
        return self.w * self.h

    def contains_point(self, px: float, py: float, pad: float = 0.0) -> bool:
        return (
            self.x - pad <= px <= self.x + self.w + pad
            and self.y - pad <= py <= self.y + self.h + pad
        )

    def iou(self, other: BBox) -> float:
        ix = max(0, min(self.x + self.w, other.x + other.w) - max(self.x, other.x))
        iy = max(0, min(self.y + self.h, other.y + other.h) - max(self.y, other.y))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union else 0.0

    def as_list(self) -> list[int]:
        return [self.x, self.y, self.w, self.h]


@dataclass
class BarcodeRead:
    """A single decoded symbol."""

    text: str
    format: str
    bbox: BBox
    quad: list[list[int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "format": self.format,
            "bbox": self.bbox.as_list(),
            "quad": self.quad,
        }


@dataclass
class ColorRead:
    """Result of classifying a tag's field color."""

    name: str
    acuity_candidates: list[Acuity]
    score: float
    coverage: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "acuity_candidates": [a.value for a in self.acuity_candidates],
            "score": round(self.score, 4),
            "coverage": round(self.coverage, 4),
        }


@dataclass
class TagDetection:
    """One triage tag found in an image."""

    patient_id: str | None
    acuity: Acuity
    confidence: float
    bbox: BBox
    color: ColorRead | None = None
    barcode: BarcodeRead | None = None
    banner_text: str | None = None
    warnings: list[str] = field(default_factory=list)
    # Where the patient ID came from: "barcode" (decoded symbol, exact),
    # "ocr" (read off the printed line -- may contain character errors),
    # or None when no ID was recovered.
    id_source: str | None = None
    # How the tag was located: "barcode" or "text".
    found_by: str = "barcode"

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "acuity": self.acuity.value,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox.as_list(),
            "color": self.color.to_dict() if self.color else None,
            "barcode": self.barcode.to_dict() if self.barcode else None,
            "banner_text": self.banner_text,
            "id_source": self.id_source,
            "found_by": self.found_by,
            "warnings": self.warnings,
        }


@dataclass
class ImageQuality:
    """How readable this frame was, for deciding whether to ask for a retake.

    Judged on OUTCOMES rather than on a picture-quality proxy. Measured across
    real frames, generic sharpness metrics did not predict success: one sheet
    read 15/15 after being blurred well below the sharpness of another that
    read 7/15. What does predict it is what actually happened -- how many
    located tags gave up a decodable barcode, and how many IDs anything
    corroborated.
    """

    rating: str                    # "good" | "marginal" | "poor" | "empty"
    barcode_decode_rate: float     # located tags whose barcode decoded
    id_verified_rate: float        # tags whose ID was independently confirmed
    tags_found: int
    sharpness: float               # informational only; NOT the rating basis
    retake_recommended: bool
    advice: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rating": self.rating,
            "retake_recommended": self.retake_recommended,
            "barcode_decode_rate": round(self.barcode_decode_rate, 3),
            "id_verified_rate": round(self.id_verified_rate, 3),
            "tags_found": self.tags_found,
            "sharpness": round(self.sharpness, 1),
            "advice": self.advice,
        }


@dataclass
class DetectionResult:
    """Everything found in one image."""

    tags: list[TagDetection]
    image_size: tuple[int, int]
    elapsed_ms: float
    warnings: list[str] = field(default_factory=list)
    quality: "ImageQuality | None" = None
    # Set when the preflight check stopped processing. `tags` is empty and the
    # frame should be retaken; nothing was missed by the abort.
    aborted: bool = False
    preflight: Any = None

    @property
    def tag_count(self) -> int:
        """Number of tags reported -- i.e. the number of line items.

        One entry per physical tag read. Two tags carrying the same patient ID
        count as two, whether or not their acuities match: reconciling that is
        a downstream decision, and collapsing them here would hide a tag that
        physically exists.
        """
        return len(self.tags)

    @property
    def identified_count(self) -> int:
        """How many of those tags yielded a patient ID."""
        return sum(1 for t in self.tags if t.patient_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tags": [t.to_dict() for t in self.tags],
            "count": self.tag_count,
            "identified_count": self.identified_count,
            "image_size": {"width": self.image_size[0], "height": self.image_size[1]},
            "elapsed_ms": round(self.elapsed_ms, 2),
            "image_quality": self.quality.to_dict() if self.quality else None,
            "preflight": self.preflight.to_dict() if self.preflight else None,
            "aborted": self.aborted,
            "warnings": self.warnings,
        }

    def roster(self) -> list[dict[str, str]]:
        """Minimal downstream payload: id + acuity, one entry per tag.

        Repeats are preserved. If the same patient ID appears on two tags, both
        appear here -- as they would if the IDs differed.
        """
        return [
            {"patient_id": t.patient_id, "acuity": t.acuity.value}
            for t in self.tags
            if t.patient_id
        ]


__all__ = [
    "Acuity",
    "BBox",
    "BarcodeRead",
    "ColorRead",
    "TagDetection",
    "DetectionResult",
    "asdict",
]
