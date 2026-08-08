"""triagevision -- read patient ID and acuity from photographs of triage tags.

    from triagevision import TriageTagDetector

    detector = TriageTagDetector()            # build once, reuse
    result = detector.detect("scene.jpg")     # path, bytes, or numpy BGR array
    result.roster()
    # [{'patient_id': 'DRILL7001', 'acuity': 'IMMEDIATE'}, ...]
"""

from .config import ColorBand, DetectorConfig
from .detector import TriageTagDetector, annotate, load_image
from .ocr import NullReader, TesseractReader, TextReader, get_reader
from .types import (
    Acuity,
    BarcodeRead,
    BBox,
    ColorRead,
    DetectionResult,
    TagDetection,
)

__version__ = "0.5.0"

__all__ = [
    "Acuity",
    "BBox",
    "BarcodeRead",
    "ColorBand",
    "ColorRead",
    "DetectionResult",
    "DetectorConfig",
    "NullReader",
    "TagDetection",
    "TesseractReader",
    "TextReader",
    "TriageTagDetector",
    "annotate",
    "get_reader",
    "load_image",
    "__version__",
]
