"""Unit tests, plus an integration test gated on a real sample image.

Run the integration test with a ground-truth sheet:

    TRIAGEVISION_SAMPLE=/path/to/photo.jpg \
    TRIAGEVISION_TRUTH='{"EA1568511":"IMMEDIATE", ...}' \
    pytest tests/
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from triagevision import Acuity, DetectorConfig, TriageTagDetector
from triagevision.barcode import _close, _merge
from triagevision.config import DEFAULT_TEXT_KEYWORDS
from triagevision.geometry import (
    normalize_landscape,
    quad_angle,
    rrect_from_quad,
    warp_upright,
)
from triagevision.ocr import match_keyword
from triagevision.types import BBox, BarcodeRead, ColorRead, TagDetection


# --------------------------------------------------------------- keyword match


@pytest.mark.parametrize(
    "text,expected",
    [
        ("IMMEDIATE", Acuity.IMMEDIATE),
        ("  delayed \n", Acuity.DELAYED),
        ("MINOR", Acuity.MINOR),
        ("EXPECTANT", Acuity.EXPECTANT),
        ("DEAD", Acuity.DEAD),
        ("MORGUE", Acuity.MORGUE),
    ],
)
def test_exact_keywords(text, expected):
    v = match_keyword(text, DEFAULT_TEXT_KEYWORDS)
    assert v.acuity is expected
    assert v.exact


def test_dead_and_morgue_stay_distinct():
    """Same clinical outcome, two protocols -- but the consuming system needs to
    know which tag it saw, so these must never be collapsed.
    """
    assert match_keyword("DEAD", DEFAULT_TEXT_KEYWORDS).acuity is Acuity.DEAD
    assert match_keyword("MORGUE", DEFAULT_TEXT_KEYWORDS).acuity is Acuity.MORGUE
    assert Acuity.DEAD is not Acuity.MORGUE
    assert Acuity.DEAD.is_deceased and Acuity.MORGUE.is_deceased
    assert not Acuity.IMMEDIATE.is_deceased


def test_vocabulary_is_exactly_the_printed_set():
    assert set(DEFAULT_TEXT_KEYWORDS) == {
        "IMMEDIATE", "DELAYED", "MINOR", "EXPECTANT", "DEAD", "MORGUE",
    }


@pytest.mark.parametrize("text,expected", [("IMMEDIAIE", Acuity.IMMEDIATE),
                                           ("DELAYFD", Acuity.DELAYED),
                                           ("MINDR", Acuity.MINOR)])
def test_fuzzy_keywords_recover_ocr_errors(text, expected):
    v = match_keyword(text, DEFAULT_TEXT_KEYWORDS)
    assert v.acuity is expected
    assert not v.exact


@pytest.mark.parametrize(
    "garbage", ["ESM STRUREARY", "PRYYSVRA0A0A", "H EASS", "AYWD Y PHS XA", ""]
)
def test_garbage_does_not_invent_a_category(garbage):
    """The safety property: OCR noise must never produce a confident acuity.

    A wrong triage category is far more dangerous than an unread one, and short
    keywords like DEAD are easy to hit by chance without a length guard.
    """
    assert match_keyword(garbage, DEFAULT_TEXT_KEYWORDS).acuity is None


def test_long_smear_cannot_match_short_keyword():
    assert match_keyword("KCRANAAAALAUTAATTATAA", DEFAULT_TEXT_KEYWORDS).acuity is None


@pytest.mark.parametrize("noise", ["DEED", "READ", "BEAD", "HEAD", "DEAO", "OEAD"])
def test_dead_is_exact_match_only(noise):
    """DEAD is four characters, where a genuine OCR slip (DEAO, OEAD) scores
    exactly the same 0.750 as unrelated real words (READ, HEAD, BEAD, DEED).
    No threshold separates them, so nothing but an exact read is accepted --
    a misread becomes UNKNOWN, which is the right outcome for the category
    that is worst to invent.
    """
    assert match_keyword(noise, DEFAULT_TEXT_KEYWORDS).acuity is None


def test_minor_still_tolerates_a_single_slip():
    """Five characters is long enough to separate a slip (0.800) from the
    nearest confusable real word, MAJOR (0.600).
    """
    assert match_keyword("MINDR", DEFAULT_TEXT_KEYWORDS).acuity is Acuity.MINOR
    assert match_keyword("MAJOR", DEFAULT_TEXT_KEYWORDS).acuity is None


def test_morgue_tolerates_a_single_slip():
    assert match_keyword("MORGVE", DEFAULT_TEXT_KEYWORDS).acuity is Acuity.MORGUE


# -------------------------------------------------------------------- geometry


def test_rrect_from_quad_uses_width_not_height():
    """Symbol height is unreliable from 1D decoders, so it must not affect the
    inferred tag box -- two quads of the same width give the same tag size.
    """
    wide_short = [[0, 0], [200, 0], [200, 6], [0, 6]]
    wide_tall = [[0, 0], [200, 0], [200, 130], [0, 130]]
    a = rrect_from_quad(wide_short)
    b = rrect_from_quad(wide_tall)
    assert a[1] == pytest.approx(b[1], rel=1e-6)
    assert a[1][0] == pytest.approx(200 * 1.45, rel=1e-6)


def test_quad_angle_is_full_360_not_an_axis():
    """A tag rotated 180 degrees must give an angle 180 degrees apart.

    This is the whole basis of orientation handling: a min-area rectangle is
    only defined up to 90 degrees and cannot distinguish a tag from the same tag
    upside down, whereas the symbol's reading direction can.
    """
    right = [[0, 0], [100, 0], [100, 20], [0, 20]]
    left = [[100, 0], [0, 0], [0, 20], [100, 20]]
    assert quad_angle(right) == pytest.approx(0.0)
    assert abs(quad_angle(left)) == pytest.approx(180.0)


@pytest.mark.parametrize("angle", [0, 45, 90, 135, 180, -90, -37])
def test_tag_box_follows_barcode_rotation(angle):
    """The inferred tag box must rotate with the symbol, and always place the
    banner on the same side of it -- for any rotation, not just cardinal ones.
    """
    rad = math.radians(angle)
    length = 100.0
    x1, y1 = length * math.cos(rad), length * math.sin(rad)
    quad = [[0, 0], [x1, y1], [x1, y1], [0, 0]]

    (cx, cy), (w, h), out_angle = rrect_from_quad(quad)
    assert w == pytest.approx(length * 1.45, rel=1e-6)
    assert out_angle == pytest.approx(angle, abs=1e-6)

    # The centre must sit perpendicular to the bars, on the banner side.
    mid = (x1 / 2.0, y1 / 2.0)
    offset = (cx - mid[0], cy - mid[1])
    along = offset[0] * math.cos(rad) + offset[1] * math.sin(rad)
    across = -offset[0] * math.sin(rad) + offset[1] * math.cos(rad)
    assert along == pytest.approx(0.0, abs=1e-6)
    assert across < 0  # banner is "up" in the tag's own frame


def test_normalize_landscape_swaps_portrait():
    (_, (w, h), _) = normalize_landscape(((10, 10), (40, 90), 0.0))
    assert (w, h) == (90, 40)


def test_warp_upright_returns_landscape_crop_and_matrix():
    img = np.zeros((200, 300, 3), np.uint8)
    out = warp_upright(img, ((150, 100), (60, 120), 30.0))
    assert out is not None
    crop, m = out
    assert crop.shape[1] >= crop.shape[0]
    assert m.shape == (3, 3)


def test_warp_upright_rejects_degenerate_rect():
    assert warp_upright(np.zeros((50, 50, 3), np.uint8), ((5, 5), (2, 2), 0.0)) is None


# --------------------------------------------------------------------- barcode


def _read(text, x, y, w=200, h=40):
    return BarcodeRead(text=text, format="Code39", bbox=BBox(x, y, w, h))


def test_merge_collapses_one_symbol_found_twice():
    """The same physical symbol re-found with a wobbly height is one patient."""
    found = [_read("EA1", 100, 100, 200, 60)]
    assert _merge(found, [_read("EA1", 103, 130, 200, 6)]) == 0
    assert len(found) == 1


def test_merge_keeps_same_id_far_apart():
    """Two tags carrying one ID is a real problem that must reach the operator."""
    found = [_read("EA1", 100, 100)]
    assert _merge(found, [_read("EA1", 1500, 1900)]) == 1
    assert len(found) == 2


def test_close_uses_long_dimension():
    """Tolerance scales with the symbol's width, so the wildly varying reported
    height cannot make one barcode look like two. Numbers here are the real
    case: a 735px-wide symbol re-found 37px away with heights 65 and 6.
    """
    assert _close(_read("A", 0, 0, 735, 65), _read("A", 0, 37, 735, 6))
    # ...but a genuinely separate tag a full symbol-width away is not merged.
    assert not _close(_read("A", 0, 0, 735, 65), _read("A", 0, 400, 735, 65))


# ------------------------------------------------------------------ decisions


def _detector(**kw):
    return TriageTagDetector(DetectorConfig(ocr_backend="none", **kw))


def test_color_alone_stays_below_review_threshold():
    """A color-only read must not present as trustworthy."""
    from triagevision.ocr import TextVerdict

    score = TriageTagDetector._score(
        has_id=True,
        verdict=TextVerdict("", None, 0.0),
        corroboration="none",
        cross_check="none",
        color=ColorRead("red", [Acuity.IMMEDIATE], score=1.0, coverage=1.0),
        located_by_color=True,
    )
    assert score < 0.60


def test_text_overrides_disagreeing_color():
    from triagevision.ocr import TextVerdict

    det = _detector(use_color=True)
    verdict = TextVerdict("IMMEDIATE", Acuity.IMMEDIATE, 1.0, exact=True)
    color = ColorRead("yellow", [Acuity.DELAYED], score=0.9, coverage=0.5)
    acuity, warns, corroboration = det._decide_acuity(verdict, color)
    assert acuity is Acuity.IMMEDIATE
    assert corroboration == "disagree"
    assert any("lighting-dependent" in w for w in warns)


def test_ambiguous_black_field_without_text_is_unknown():
    """Most vendors print DEAD and EXPECTANT on one black field; color cannot
    separate them, so an unreadable banner must yield UNKNOWN, not a coin flip.
    """
    from triagevision.ocr import TextVerdict

    det = _detector(use_color=True)
    color = ColorRead("black", [Acuity.DEAD, Acuity.EXPECTANT], score=1.0, coverage=0.9)
    acuity, warns, _ = det._decide_acuity(TextVerdict("", None, 0.0), color)
    assert acuity is Acuity.UNKNOWN
    assert any("manual review" in w for w in warns)


def test_require_text_refuses_color_inference():
    from triagevision.ocr import TextVerdict

    det = _detector(use_color=True, require_text=True)
    color = ColorRead("red", [Acuity.IMMEDIATE], score=1.0, coverage=0.9)
    acuity, _, _ = det._decide_acuity(TextVerdict("", None, 0.0), color)
    assert acuity is Acuity.UNKNOWN


def test_weak_text_with_no_corroboration_is_unknown():
    from triagevision.ocr import TextVerdict

    det = _detector()
    weak = TextVerdict("DEED", Acuity.DEAD, 0.74)
    acuity, _, _ = det._decide_acuity(weak, None)
    assert acuity is Acuity.UNKNOWN


def _tag(pid, x, y, conf=0.9):
    return TagDetection(
        patient_id=pid, acuity=Acuity.IMMEDIATE, confidence=conf,
        bbox=BBox(x, y, 100, 50),
    )


def test_overlapping_duplicates_collapse_to_best():
    kept = TriageTagDetector._merge_same_tag(
        [_tag("EA1", 0, 0, 0.6), _tag("EA1", 5, 5, 0.95)]
    )
    assert len(kept) == 1
    assert kept[0].confidence == 0.95


def test_distant_duplicates_are_kept_and_flagged():
    tags = [_tag("EA1", 0, 0), _tag("EA1", 900, 900)]
    kept = TriageTagDetector._merge_same_tag(tags)
    assert len(kept) == 2
    assert TriageTagDetector._duplicate_ids(kept) == {"EA1"}


# ----------------------------------------------------------------- config/IO


def test_id_pattern_rejects_junk():
    cfg = DetectorConfig()
    assert cfg.is_valid_patient_id("EA1568511")
    assert not cfg.is_valid_patient_id("A")
    assert not cfg.is_valid_patient_id("")


def test_detect_on_blank_image_returns_nothing():
    result = _detector().detect(np.full((600, 800, 3), 128, np.uint8))
    assert result.tags == []
    assert result.image_size == (800, 600)


def test_color_disabled_by_default():
    assert DetectorConfig().use_color is False


# --------------------------------------------------------------- integration


@pytest.mark.skipif(
    not os.getenv("TRIAGEVISION_SAMPLE"), reason="set TRIAGEVISION_SAMPLE to run"
)
def test_sample_sheet_matches_ground_truth():
    truth = json.loads(os.environ["TRIAGEVISION_TRUTH"])
    result = TriageTagDetector().detect(os.environ["TRIAGEVISION_SAMPLE"])
    got = {t.patient_id: t.acuity.value for t in result.tags if t.patient_id}
    assert got == truth
