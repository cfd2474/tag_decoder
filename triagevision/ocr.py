"""Banner-text reading -- the PRIMARY acuity signal.

Why text leads and color follows: the printed word is invariant. The field color
is not. A red tag photographed under sodium-vapor light, a scene light, an IR
illuminator, or through smoke can land anywhere from orange to brown in HSV, and
a green tag under warm light drifts toward yellow -- which is a one-category
triage error in the direction that matters. The word IMMEDIATE is still the word
IMMEDIATE. So color is used to corroborate and to fill gaps, never to override a
confident read.

OCR is still NOT used for the patient ID. The barcode gives that exactly, and
OCR on alphanumeric IDs is the biggest source of silent error in a pipeline like
this (0/O, 1/I/l, 5/S, 8/B, 2/Z). Text handles the closed five-word vocabulary,
where fuzzy matching makes a sloppy read recoverable; the barcode handles the
open-vocabulary field where it would not be.

Backends are pluggable. Implement `TextReader` to swap in PaddleOCR, EasyOCR,
Apple Vision, or a cloud endpoint.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Iterator, Protocol

import cv2
import numpy as np

from .types import Acuity


# How far ahead of the runner-up a fuzzy match must sit to count as confident.
#
# Measured over real banner reads and real OCR garbage from the sample sheets,
# the two populations separate completely on margin, and only partially on
# absolute score:
#
#   real reads   score 0.696-0.933   margin 0.267-0.600
#   garbage      score 0.190-0.444   margin 0.000-0.111
#
# Absolute score alone is the wrong discriminator. "XEECTANTAY" scores just
# 0.737 against EXPECTANT -- OCR chewed both ends -- but scores 0.353 against
# everything else, so the category is not in doubt. Garbage scores poorly
# against every word roughly equally, which is exactly what a small margin
# means. This is the same idea as a nearest-neighbour ratio test.
MIN_MARGIN = 0.20


@dataclass
class TextVerdict:
    """Outcome of reading one region."""

    text: str
    acuity: Acuity | None
    score: float          # 0-1 similarity to the matched keyword
    exact: bool = False   # keyword found verbatim, not fuzzily
    margin: float = 0.0   # lead over the next-best keyword

    @property
    def confident(self) -> bool:
        """Whether this match can stand on its own, with nothing corroborating.

        Judged on the lead over the runner-up rather than the raw score. An
        absolute threshold rejected legitimate reads of long words whose ends
        were garbled, while admitting nothing extra -- and since field colour is
        off by default, "weak unless colour agrees" meant "weak, always".
        """
        return self.acuity is not None and (self.exact or self.margin >= MIN_MARGIN)


class TextReader(Protocol):
    """Anything that turns an image crop into a text string."""

    available: bool

    def read(self, crop: np.ndarray) -> str: ...


class NullReader:
    """No OCR installed. The pipeline falls back to color and says so."""

    available = False

    def read(self, crop: np.ndarray) -> str:  # noqa: ARG002
        return ""


class TesseractReader:
    """pytesseract backend. Needs the `tesseract` binary plus `pip install pytesseract`.

    Whitelisted to A-Z and a space: triage banners are uppercase words, and
    dropping digits/punctuation stops the patient ID bleeding into the read when
    the ROI is generous.
    """

    # psm 7 = one text line (a tight banner crop); psm 6 = uniform block (a
    # looser crop that also catches the ID line and the top of the barcode,
    # which is the common case). psm 11 (sparse text) is omitted: it almost
    # never won, and under a call budget it only displaced a mode that would.
    PSM_MODES = (7, 6)

    def __init__(self, lang: str = "eng", psm_modes: tuple[int, ...] | None = None):
        self.lang = lang
        self.psm_modes = psm_modes or self.PSM_MODES
        try:
            import pytesseract  # noqa: PLC0415

            pytesseract.get_tesseract_version()
            self._pt = pytesseract
            self.available = True
        except Exception:
            self._pt = None
            self.available = False

    ID_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"

    def _cfg(self, psm: int, whitelist: str | None = None) -> str:
        chars = whitelist or "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return (
            f"--oem 1 --psm {psm} "
            f"-c tessedit_char_whitelist={chars} "
            "-c load_system_dawg=0 -c load_freq_dawg=0"
        )

    def read_id(self, crop: np.ndarray) -> list[str]:
        """Read the human-readable ID line, returning candidate tokens.

        This is a CROSS-CHECK on the barcode, not a replacement for it -- see
        the module docstring on why OCR must not be trusted with an open
        vocabulary. It earns its place because a Code 39 symbol without a check
        digit can decode cleanly to the wrong string, and comparing against the
        line printed beside it is the only cheap way to notice.
        """
        if not self.available or crop is None or crop.size == 0:
            return []
        tokens: list[str] = []
        # Both polarities, in single-line AND block mode. Block mode matters:
        # the ID band inevitably catches part of the banner above and the
        # barcode below, and single-line mode cannot parse three competing rows
        # -- restricting to it drops the match rate from 4/5 to 2/5 on a real
        # sheet. This is the only independent check on a patient ID when the
        # symbology has no check character, so it is worth the extra ~0.3s.
        for prepped in list(preprocess_variants(crop))[:2]:
            for psm in (7, 6):
                try:
                    raw = self._pt.image_to_string(
                        prepped, lang=self.lang,
                        config=self._cfg(psm, self.ID_CHARS),
                    )
                except Exception:
                    continue
                for tok in raw.replace("\n", " ").split():
                    tok = tok.strip("-")
                    if len(tok) >= 4 and tok not in tokens:
                        tokens.append(tok)
        return tokens

    def read(self, crop: np.ndarray) -> str:
        if not self.available or crop.size == 0:
            return ""
        try:
            return self._pt.image_to_string(crop, lang=self.lang,
                                            config=self._cfg(self.psm_modes[0])).strip()
        except Exception:
            return ""

    def read_variants(self, crop: np.ndarray) -> Iterator[str]:
        """Yield reads across (page-segmentation x preprocessing), best bet first.

        A generator, not a list, so the caller stops the moment it has an exact
        keyword hit -- which for a clean banner is the very first call.

        Preprocessing is the OUTER loop and segmentation mode the inner one, so
        a small budget covers BOTH unknowns rather than exhausting itself on
        one. Two things decide success independently: polarity (white-on-red vs
        black-on-yellow) and whether the crop holds one text line or several.
        Ordering by segmentation first meant a three-call budget spent every
        call on single-line mode and never reached block mode -- which silently
        lost the acuity on perfectly legible IMMEDIATE banners, because their
        band also contained the ID line and the top of the barcode.

        Each call shells out to the tesseract binary at ~100-200ms, so this
        ordering is the difference between reading a tag and not.
        """
        if not self.available or crop.size == 0:
            return
        for prepped in preprocess_variants(crop):
            for psm in self.psm_modes:
                try:
                    t = self._pt.image_to_string(
                        prepped, lang=self.lang, config=self._cfg(psm)
                    ).strip()
                except Exception:
                    continue
                if t:
                    yield t


def get_reader(backend: str) -> TextReader:
    """Resolve a backend name. "auto" prefers tesseract, degrades to null."""
    backend = (backend or "auto").lower()
    if backend in ("none", "null", "off"):
        return NullReader()
    if backend in ("tesseract", "auto"):
        reader = TesseractReader()
        if reader.available or backend == "tesseract":
            return reader
        return NullReader()
    raise ValueError(f"unknown ocr backend: {backend!r}")


# --------------------------------------------------------------------- imaging


def _resize_for_ocr(gray: np.ndarray, target_h: int = 64, max_w: int = 700) -> np.ndarray:
    """Bring a crop into the size band tesseract works best and fastest in.

    Upscale tiny crops toward a ~30px cap height, and -- just as important --
    downscale big ones. A 1200px-wide banner carries no more information than a
    700px one for a word this large, but costs several times as much per call,
    and every call shells out to the tesseract binary.
    """
    h, w = gray.shape[:2]
    if h < target_h:
        s = target_h / max(h, 1)
        return cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    if w > max_w:
        s = max_w / float(w)
        return cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return gray


def preprocess_variants(crop: np.ndarray) -> Iterator[np.ndarray]:
    """Yield binarizations of one crop, in both polarities, easiest case first.

    Triage banners come in both polarities -- white on red/green, black on
    yellow -- and we cannot know which without trusting color, which is exactly
    the dependency we are trying to remove. So both are produced and the keyword
    match picks the winner. Lazily, so a clean tag costs one threshold op.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = _resize_for_ocr(gray)
    gray = cv2.bilateralFilter(gray, 5, 60, 60)

    def framed(v: np.ndarray) -> np.ndarray:
        # Tesseract wants quiet space around the text.
        return cv2.copyMakeBorder(v, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255)

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield framed(otsu)
    yield framed(cv2.bitwise_not(otsu))

    # CLAHE + Otsu recovers washed-out, backlit or under-exposed banners.
    eq = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    _, eq_otsu = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield framed(eq_otsu)
    yield framed(cv2.bitwise_not(eq_otsu))

    # Adaptive handles uneven lighting across a large or curled tag.
    blk = max(15, (min(gray.shape) // 4) | 1)
    adap = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blk, 10
    )
    yield framed(adap)
    yield framed(cv2.bitwise_not(adap))


# -------------------------------------------------------------------- matching


def _fuzzy_floor(word: str, min_ratio: float) -> float:
    """Similarity a token must reach to be accepted as `word`.

    Short words tolerate less fuzz, because for them a genuine OCR slip and an
    entirely different word are indistinguishable. Measured against this
    vocabulary:

      DEAD (4)   single-char slips DEAO/OEAD score 0.750 -- and so do READ,
                 HEAD, BEAD and DEED. There is no threshold that admits the
                 slips while rejecting the real words, so DEAD is EXACT-ONLY.
                 A misread yields UNKNOWN and a human look, which is the right
                 outcome for the category that is worst to invent.
      MINOR (5)  slips score 0.800, while the nearest confusable real word
                 (MAJOR) scores 0.600. Separable, so 0.78.
      6+         slips score 0.833 or better; the general floor holds.
    """
    if len(word) <= 4:
        return 1.01  # unreachable by definition: exact match only
    if len(word) == 5:
        return 0.78
    return min_ratio


def match_keyword(
    text: str, keywords: dict[str, Acuity], min_ratio: float = 0.72
) -> TextVerdict:
    """Map a raw OCR string onto the closed acuity vocabulary.

    Exact substring first (certain), then fuzzy per-token. The vocabulary is
    five words sharing almost no letters, so a mangled read still lands on the
    right one -- "IMMEDIAIE", "DELAYFD" and "MINDR" all resolve.

    Two guards keep noise from inventing a category. Tokens must be within a
    plausible length of the word they match, and the similarity floor is high.
    Without them, OCR garbage off a badly-cropped tag scores just high enough
    against a short keyword to be accepted -- which is how a string like
    "ESM STRUREARY" gets read as DEAD. On a triage tag a confidently wrong
    category is far worse than an honest UNKNOWN.
    """
    if not text:
        return TextVerdict("", None, 0.0)

    upper = "".join(c if c.isalpha() or c.isspace() else " " for c in text.upper())
    upper = " ".join(upper.split())
    if not upper:
        return TextVerdict(text, None, 0.0)

    for word, acuity in keywords.items():
        if word in upper:
            return TextVerdict(text, acuity, 1.0, exact=True, margin=1.0)

    tokens = [t for t in upper.split() if len(t) >= 4]
    if upper not in tokens and len(upper) < 24:
        tokens.append(upper)

    best_acuity: Acuity | None = None
    best_ratio = 0.0
    best_margin = 0.0

    for token in tokens:
        # Score against every word first, so the runner-up is known. The margin
        # over it is what distinguishes a mangled real read from noise: garbage
        # sits roughly equidistant from the whole vocabulary.
        ratios = {
            word: difflib.SequenceMatcher(None, token, word).ratio()
            for word in keywords
        }
        for word, acuity in keywords.items():
            if abs(len(token) - len(word)) > max(2, int(0.35 * len(word))):
                continue
            ratio = ratios[word]
            if ratio < _fuzzy_floor(word, min_ratio):
                continue
            runner_up = max(
                (v for w, v in ratios.items() if w != word), default=0.0
            )
            margin = ratio - runner_up
            # Prefer the clearest discrimination, breaking ties on similarity.
            if (margin, ratio) > (best_margin, best_ratio) or best_acuity is None:
                best_ratio, best_margin = ratio, margin
                best_acuity = acuity

    if best_acuity is None:
        return TextVerdict(text, None, best_ratio)
    return TextVerdict(text, best_acuity, best_ratio, margin=best_margin)


def best_verdict(
    reader: TextReader,
    crops: list[np.ndarray],
    keywords: dict[str, Acuity],
    max_calls: int = 4,
) -> TextVerdict:
    """Read candidate ROIs and keep the strongest keyword match.

    Stops early on an exact hit, and hard-caps total engine calls. The cap is
    what keeps a frame full of unreadable tags from degrading into a minute of
    OCR: a tag that has not resolved within a few attempts is not going to,
    and the honest answer for it is a low-confidence flag for human review.
    """
    best = TextVerdict("", None, 0.0)
    if not getattr(reader, "available", False):
        return best

    read_many = getattr(reader, "read_variants", None)
    calls = 0
    for crop in crops:
        if crop is None or crop.size == 0:
            continue
        texts = read_many(crop) if read_many else [reader.read(crop)]
        for t in texts:
            calls += 1
            v = match_keyword(t, keywords)
            if v.exact:
                return v
            if (v.margin, v.score) > (best.margin, best.score):
                best = v
            if calls >= max_calls:
                return best
    return best
