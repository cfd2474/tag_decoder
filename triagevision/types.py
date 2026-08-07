"""Core data types for triage tag detection."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, asdict
from typing import Any


class Acuity(str, enum.Enum):
    """START/SALT triage categories."""

    IMMEDIATE = "IMMEDIATE"
    DELAYED = "DELAYED"
    MINOR = "MINOR"
    EXPECTANT = "EXPECTANT"
    DEAD = "DEAD"
    UNKNOWN = "UNKNOWN"


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "acuity": self.acuity.value,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox.as_list(),
            "color": self.color.to_dict() if self.color else None,
            "barcode": self.barcode.to_dict() if self.barcode else None,
            "banner_text": self.banner_text,
            "warnings": self.warnings,
        }


@dataclass
class DetectionResult:
    """Everything found in one image."""

    tags: list[TagDetection]
    image_size: tuple[int, int]
    elapsed_ms: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tags": [t.to_dict() for t in self.tags],
            "count": len(self.tags),
            "image_size": {"width": self.image_size[0], "height": self.image_size[1]},
            "elapsed_ms": round(self.elapsed_ms, 2),
            "warnings": self.warnings,
        }

    def roster(self) -> list[dict[str, str]]:
        """Minimal downstream payload: just id + acuity."""
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
