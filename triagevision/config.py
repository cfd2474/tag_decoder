"""Tunable configuration: color profiles, geometry gates, ID validation.

Everything that varies by jurisdiction, tag vendor, or camera lives here so the
detection code stays generic. Load overrides from JSON with `DetectorConfig.from_json`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .types import Acuity


@dataclass(frozen=True)
class ColorBand:
    """One color class, expressed as one or more HSV ranges (OpenCV convention:
    H in 0-179, S/V in 0-255). Multiple ranges let red wrap the hue origin.
    """

    name: str
    hsv_ranges: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]
    acuity_candidates: tuple[Acuity, ...]
    # Minimum fraction of the tag's non-white pixels that must match to accept.
    min_coverage: float = 0.10


# START / METTAG default scheme. Black is deliberately ambiguous: most tag
# vendors use one black field for both DEAD and EXPECTANT, so color alone cannot
# separate them -- the banner text does. See `ambiguity` handling in detector.py.
DEFAULT_COLOR_BANDS: tuple[ColorBand, ...] = (
    ColorBand(
        name="red",
        hsv_ranges=(
            ((0, 90, 70), (10, 255, 255)),
            ((170, 90, 70), (179, 255, 255)),
        ),
        acuity_candidates=(Acuity.IMMEDIATE,),
    ),
    ColorBand(
        name="yellow",
        hsv_ranges=(((18, 90, 110), (36, 255, 255)),),
        acuity_candidates=(Acuity.DELAYED,),
    ),
    ColorBand(
        name="green",
        hsv_ranges=(((40, 60, 50), (88, 255, 255)),),
        acuity_candidates=(Acuity.MINOR,),
    ),
    ColorBand(
        name="slate",
        # EXPECTANT stock is described as "grey" but is often printed slate
        # blue: measured on a real tag, the field sits at H=107 S=82 V=137,
        # which a neutral-grey band (low saturation only) catches by accident
        # at best. Both are covered.
        hsv_ranges=(
            ((0, 0, 60), (179, 45, 190)),      # neutral grey
            ((90, 30, 60), (130, 150, 205)),   # slate blue, as measured
        ),
        acuity_candidates=(Acuity.EXPECTANT,),
        min_coverage=0.15,
    ),
    ColorBand(
        name="black",
        # DEAD and MORGUE print on the same black field, so colour cannot tell
        # them apart -- only the banner word can. With the word unreadable this
        # resolves to UNKNOWN rather than guessing between two protocols.
        hsv_ranges=(((0, 0, 0), (179, 90, 55)),),
        acuity_candidates=(Acuity.DEAD, Acuity.MORGUE),
        min_coverage=0.15,
    ),
)


# Banner words -> acuity. This is the exact printed vocabulary of the tag stock,
# always uppercase, and it is deliberately CLOSED: every extra synonym is another
# chance for OCR noise to match something. Add words only if your stock prints
# them.
#
# DEAD and MORGUE map to distinct categories on purpose. They are the same
# clinical outcome under two different triage protocols, and the consuming
# system needs to know which tag it was looking at.
DEFAULT_TEXT_KEYWORDS: dict[str, Acuity] = {
    "IMMEDIATE": Acuity.IMMEDIATE,
    "DELAYED": Acuity.DELAYED,
    "MINOR": Acuity.MINOR,
    "EXPECTANT": Acuity.EXPECTANT,
    "DEAD": Acuity.DEAD,
    "MORGUE": Acuity.MORGUE,
}


@dataclass
class DetectorConfig:
    # --- color segmentation ---
    # OFF by default. Color is the least trustworthy signal on a triage tag and
    # under poor lighting it does active harm, not merely nothing: a white
    # barcode label under a blue scene light segments as a blue "tag", so the
    # inferred tag box collapses onto the label and the banner falls outside the
    # crop entirely -- turning a readable tag into an unreadable one. Barcodes
    # localize better in every lighting condition measured. Enable this only if
    # you need acuity for tags whose barcode cannot be decoded at all.
    use_color: bool = False
    color_bands: tuple[ColorBand, ...] = DEFAULT_COLOR_BANDS
    # Reject candidate regions smaller than this fraction of the frame.
    min_tag_area_frac: float = 0.0015
    max_tag_area_frac: float = 0.90
    # Tag aspect ratio (w/h) acceptance window, generous for perspective skew.
    min_aspect: float = 0.25
    max_aspect: float = 8.0
    # Morphological close kernel as a fraction of the image's short side.
    # Keep this SMALL. Its job is only to bridge the white dashes inside a tag's
    # border; anything larger bridges the gap *between* stacked same-color tags
    # and merges four IMMEDIATE tags into one giant region.
    close_kernel_frac: float = 0.004
    close_kernel_max_px: int = 9
    # A candidate must fill at least this much of its own bounding box.
    min_extent: float = 0.35

    # --- barcode ---
    # Try these upscale factors when the first decode pass finds nothing new.
    barcode_scales: tuple[float, ...] = (1.0, 1.5, 2.0)
    # Regex a decoded symbol must satisfy to be treated as a patient ID.
    # Default accepts most alphanumeric ID schemes; tighten for your own.
    patient_id_pattern: str = r"^[A-Za-z0-9][A-Za-z0-9._\-/]{2,63}$"
    # Padding (fraction of barcode height) when matching a barcode to a tag.
    match_pad_frac: float = 0.75

    # --- tag layout priors, used to infer a tag box from a bare barcode ---
    # Measured from standard START-scheme tags; override per vendor. All are
    # derived from the symbol's width, which decoders localize reliably, never
    # from its height, which they do not.
    tag_width_per_barcode_width: float = 1.45
    tag_aspect_prior: float = 2.25          # tag width / tag height
    barcode_offset_frac: float = 0.17       # symbol centre below tag centre, xH

    # --- text localization (finds tags the barcode decoder cannot) ---
    # Scan the frame for banner words and treat each as a tag anchor. This is
    # what keeps a soft photo usable: wide letter strokes survive blur that
    # destroys narrow bars, so a frame where most symbols fail to decode can
    # still have every banner legible. Costs ~2-4s; turn off only if every
    # frame is known to be sharp.
    use_text_localizer: bool = True
    # Width the frame is downscaled to for the banner scan. Banners are large,
    # and the engine is far faster on fewer pixels.
    text_scan_width: int = 1800
    text_scan_psms: tuple[int, ...] = (6, 11)

    # --- text (primary acuity signal) ---
    text_keywords: dict[str, Acuity] = field(
        default_factory=lambda: dict(DEFAULT_TEXT_KEYWORDS)
    )
    # OCR backend: "none" | "tesseract" | "auto"
    ocr_backend: str = "auto"
    # "always" reads the banner on every tag; "never" disables OCR entirely and
    # leaves you on the lighting-dependent color path.
    ocr_policy: str = "always"  # "never" | "always"
    # Strict mode: emit UNKNOWN rather than inferring acuity from color when the
    # banner text cannot be read. Turn this on where a wrong category is worse
    # than no category.
    require_text: bool = False

    # --- performance ---
    # Threads used to read tags within one frame. 0 = one per CPU, 1 = serial.
    # Set to 1 if you are already running several detections concurrently, so
    # the two levels of parallelism do not oversubscribe the box.
    max_workers: int = 0

    # --- output ---
    min_confidence: float = 0.25
    # Emit tags that have a color but no readable barcode.
    emit_unidentified_tags: bool = True
    # Emit barcodes that matched no colored region (tag torn/obscured).
    emit_orphan_barcodes: bool = True

    def __post_init__(self) -> None:
        self._id_re = re.compile(self.patient_id_pattern)

    def is_valid_patient_id(self, text: str) -> bool:
        return bool(self._id_re.match(text.strip()))

    @classmethod
    def from_json(cls, path: str | Path) -> DetectorConfig:
        """Load overrides. Unknown keys are ignored; color bands may be replaced
        wholesale with a list of {name, hsv_ranges, acuity_candidates} objects.
        """
        data = json.loads(Path(path).read_text())
        bands = data.pop("color_bands", None)
        cfg = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        if bands:
            cfg.color_bands = tuple(
                ColorBand(
                    name=b["name"],
                    hsv_ranges=tuple(
                        (tuple(lo), tuple(hi)) for lo, hi in b["hsv_ranges"]
                    ),
                    acuity_candidates=tuple(
                        Acuity(a) for a in b["acuity_candidates"]
                    ),
                    min_coverage=b.get("min_coverage", 0.10),
                )
                for b in bands
            )
        return cfg
