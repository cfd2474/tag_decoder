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
from triagevision.barcode import _close, _merge, is_self_checking
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


def test_expectant_band_covers_grey_and_slate_blue():
    """EXPECTANT stock is called "grey" but is often printed slate blue
    (measured: H=107 S=82 V=137). Both must fall in the band.
    """
    import cv2

    from triagevision.color import build_masks
    from triagevision.config import DEFAULT_COLOR_BANDS

    for hsv in ((107, 82, 137), (0, 0, 130)):  # slate blue, neutral grey
        patch = cv2.cvtColor(
            np.full((20, 20, 3), hsv, np.uint8), cv2.COLOR_HSV2BGR
        )
        masks = build_masks(patch, DEFAULT_COLOR_BANDS)
        assert masks["slate"].any(), f"{hsv} not matched as EXPECTANT field"


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


def test_garbled_long_word_still_resolves_confidently():
    """A real read whose ends OCR mangled must not be downgraded to UNKNOWN.

    This exact banner text came off a production server. It scores only 0.74
    against EXPECTANT, but 0.35 against everything else -- the category is not
    in doubt, and an absolute-score threshold wrongly rejected it. Confidence is
    judged on the lead over the runner-up.
    """
    v = match_keyword("BE XEECTANTAY\nSN", DEFAULT_TEXT_KEYWORDS)
    assert v.acuity is Acuity.EXPECTANT
    assert v.score < 0.80          # would fail an absolute threshold
    assert v.margin >= 0.20        # but is unambiguous
    assert v.confident


@pytest.mark.parametrize(
    "text,expected",
    [("MMEDITE", Acuity.IMMEDIATE), ("MINDR", Acuity.MINOR),
     ("MORGVE", Acuity.MORGUE), ("XEECTANTAY", Acuity.EXPECTANT)],
)
def test_mangled_reads_are_confident_without_colour(text, expected):
    """Colour is off by default, so a rule requiring colour corroboration to
    accept a fuzzy match could never be satisfied -- it meant "weak, always".
    """
    v = match_keyword(text, DEFAULT_TEXT_KEYWORDS)
    assert v.acuity is expected and v.confident


@pytest.mark.parametrize(
    "garbage",
    ["ESM STRUREARY", "SEESUEEEVEPEEB", "NGTUNN", "H EASS", "AYWD Y PHS XA"],
)
def test_garbage_has_no_margin_and_stays_rejected(garbage):
    """Noise sits roughly equidistant from the whole vocabulary; that lack of a
    lead is what separates it from a mangled real read.
    """
    v = match_keyword(garbage, DEFAULT_TEXT_KEYWORDS)
    assert v.acuity is None
    assert not v.confident


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


def test_tile_pass_maps_coordinates_back_to_the_frame(monkeypatch):
    """A hit found inside a tile must be reported in full-frame coordinates.

    Get this wrong and every tag localizes to the wrong place, which downstream
    looks like a detection failure rather than an offset bug.
    """
    from triagevision import barcode as bm

    seen: list[tuple[int, int]] = []

    def fake_decode(tile, scale, formats):
        seen.append(tile.shape[:2])
        # One hit at a fixed offset inside whatever tile we are given.
        quad = [[10, 5], [60, 5], [60, 15], [10, 15]]
        return [BarcodeRead("X", "Code 39", bm._quad_to_bbox(quad), quad)]

    monkeypatch.setattr(bm, "_decode_pass", fake_decode)
    monkeypatch.setattr(bm.geometry, "suppress_glare", lambda g: g)

    gray = np.zeros((600, 800), np.uint8)
    reads = bm._tile_pass(gray, None, grid=(2, 2), overlap=0.0)

    assert seen, "no tiles were scanned"
    # Four tiles, each contributing a hit at a different frame position.
    assert len({(r.bbox.x, r.bbox.y) for r in reads}) == 4
    # The tile at column 1 starts at x=400, so its hit lands at x=410.
    assert any(r.bbox.x == 410 for r in reads)
    assert any(r.bbox.y == 305 for r in reads)  # row 1 starts at y=300


def test_tile_grid_covers_the_whole_frame(monkeypatch):
    """No gaps between tiles: a symbol must not fall between the cracks."""
    from triagevision import barcode as bm

    boxes: list[tuple[int, int, int, int]] = []

    def record(tile, scale, formats):
        return []

    monkeypatch.setattr(bm, "_decode_pass", record)
    monkeypatch.setattr(bm.geometry, "suppress_glare", lambda g: g)

    h, w = 600, 800
    covered = np.zeros((h, w), bool)
    rows, cols = 3, 3
    overlap = 0.25
    tile_h, tile_w = h / rows, w / cols
    pad_y, pad_x = tile_h * overlap, tile_w * overlap
    for r in range(rows):
        for c in range(cols):
            y0 = max(0, int(r * tile_h - pad_y))
            y1 = min(h, int((r + 1) * tile_h + pad_y))
            x0 = max(0, int(c * tile_w - pad_x))
            x1 = min(w, int((c + 1) * tile_w + pad_x))
            covered[y0:y1, x0:x1] = True
            boxes.append((x0, y0, x1, y1))
    assert covered.all()
    # Overlap must be real, or a symbol on a seam is cut in every tile.
    assert boxes[0][2] > int(w / cols)


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


def test_two_tags_sharing_an_id_are_both_reported():
    """Duplicate patient IDs are reported, not reconciled.

    Two physical tags carrying the same ID become two line items, exactly as
    they would with different IDs -- whether or not their acuities agree.
    Deciding what a duplicate means belongs to the consuming system; hiding one
    here would conceal a tag that physically exists.
    """
    from triagevision.types import DetectionResult

    for second_acuity in (Acuity.IMMEDIATE, Acuity.DELAYED):
        a = _tag("EA1", 0, 0)
        b = _tag("EA1", 900, 900)
        b.acuity = second_acuity
        kept = TriageTagDetector._merge_same_tag([a, b])
        assert len(kept) == 2

        result = DetectionResult(tags=kept, image_size=(10, 10), elapsed_ms=0.0)
        assert result.tag_count == 2
        assert len(result.roster()) == 2
        assert [r["patient_id"] for r in result.roster()] == ["EA1", "EA1"]
        assert result.warnings == []


def test_output_schema_matches_the_documented_fields():
    """Lock the wire format against the README's "Output reference".

    Consumers parse these keys, and the docs promise exactly this set. Adding,
    renaming or dropping a field should fail here and force the README to be
    updated in the same commit, rather than silently drifting.
    """
    from triagevision.types import DetectionResult

    tag = TagDetection(
        patient_id="EA1",
        acuity=Acuity.IMMEDIATE,
        confidence=0.95,
        bbox=BBox(0, 0, 10, 10),
        color=ColorRead("red", [Acuity.IMMEDIATE], score=0.9, coverage=0.5),
        barcode=BarcodeRead("EA1", "Code 39", BBox(1, 1, 5, 2), quad=[[1, 1]] * 4),
        banner_text="IMMEDIATE",
    )
    payload = DetectionResult([tag], (100, 200), 12.5).to_dict()

    assert set(payload) == {
        "tags", "count", "identified_count", "image_size", "elapsed_ms",
        "image_quality", "warnings",
    }
    assert set(payload["image_size"]) == {"width", "height"}
    assert set(payload["tags"][0]) == {
        "patient_id", "acuity", "confidence", "bbox", "color", "barcode",
        "banner_text", "id_source", "found_by", "warnings",
    }
    assert set(payload["tags"][0]["barcode"]) == {"text", "format", "bbox", "quad"}
    assert set(payload["tags"][0]["color"]) == {
        "name", "acuity_candidates", "score", "coverage",
    }
    # roster() is the minimal contract and must stay exactly two keys.
    assert set(DetectionResult([tag], (1, 1), 0.0).roster()[0]) == {
        "patient_id", "acuity",
    }


def test_documented_acuity_values_are_the_full_set():
    assert {a.value for a in Acuity} == {
        "IMMEDIATE", "DELAYED", "MINOR", "EXPECTANT", "DEAD", "MORGUE", "UNKNOWN",
    }


def test_counts_report_line_items_and_identified():
    from triagevision.types import DetectionResult

    tags = [_tag("EA1", 0, 0), _tag("EA2", 900, 900), _tag(None, 400, 400)]
    result = DetectionResult(tags=tags, image_size=(10, 10), elapsed_ms=0.0)
    assert result.tag_count == 3
    assert result.identified_count == 2
    assert result.to_dict()["count"] == 3
    assert result.to_dict()["identified_count"] == 2


# ----------------------------------------------------------------- config/IO


def test_id_pattern_rejects_junk():
    cfg = DetectorConfig()
    assert cfg.is_valid_patient_id("EA1568511")
    assert not cfg.is_valid_patient_id("A")
    assert not cfg.is_valid_patient_id("")


@pytest.mark.parametrize(
    "serial",
    ["EA1568511", "SN1050837", "0001", "X-99-ABC", "TRIAGE/2026/0042", "9812734650"],
)
def test_id_pattern_accepts_unknown_vendor_serials(serial):
    """IDs are pre-printed vendor serials, so the next batch's format is not
    knowable. A pattern fitted to today's stock would silently drop every
    patient in a batch that looks different.
    """
    assert DetectorConfig().is_valid_patient_id(serial)


def _id_case(payload, printed):
    """Run the barcode/printed-ID reconciliation with a preset OCR result."""
    from triagevision.detector import _Candidate, _Oriented

    det = _detector()
    cand = _Candidate(
        bbox=BBox(0, 0, 100, 50),
        barcode=BarcodeRead(text=payload, format="Code 39", bbox=BBox(0, 0, 80, 20)),
    )
    oriented = _Oriented(crop=np.zeros((10, 10, 3), np.uint8), printed_id=printed)
    return det._resolve_id(cand, oriented)


def test_printed_id_confirms_only_on_exact_match():
    _, _, cross = _id_case("EA1568513", "EA1568513")
    assert cross == "agree"


def test_one_digit_off_serial_is_never_treated_as_confirmation():
    """The safety property that matters most here.

    Patient IDs are sequential vendor serials, so the likeliest barcode misread
    is one digit -- and EA1568513 vs EA1568512 scores 0.89 similarity. Any
    threshold loose enough to absorb OCR noise would bless exactly the error
    this cross-check exists to catch, so confirmation demands an exact match.
    """
    _, warns, cross = _id_case("EA1568513", "EA1568512")
    assert cross != "agree"
    assert any("unverified" in w for w in warns)


def test_ocr_noise_on_printed_line_is_not_a_false_alarm():
    """A garbled printed read is not evidence against the barcode. Crying
    mismatch here trains operators to ignore the warnings that do matter.
    """
    _, warns, cross = _id_case("EA1568513", "11568575")
    assert cross == "none"
    assert not any("verify this tag" in w for w in warns)


def test_genuinely_different_string_is_flagged():
    _, warns, cross = _id_case("EA1568513", "ZZ9990001")
    assert cross == "disagree"
    assert any("verify this tag" in w for w in warns)


def test_unverified_warning_only_for_non_self_checking_symbologies():
    from triagevision.detector import _Candidate, _Oriented

    det = _detector()
    oriented = _Oriented(crop=np.zeros((10, 10, 3), np.uint8), printed_id=None)
    for fmt, expect_warning in (("Code 39", True), ("Code 128", False)):
        cand = _Candidate(
            bbox=BBox(0, 0, 100, 50),
            barcode=BarcodeRead(text="EA1", format=fmt, bbox=BBox(0, 0, 80, 20)),
        )
        _, warns, _ = det._resolve_id(cand, oriented)
        assert any("unverified" in w for w in warns) is expect_warning


@pytest.mark.parametrize(
    "fmt,expected",
    [("Code 39", False), ("Codabar", False), ("ITF", False),
     ("Code 128", True), ("QRCode", True), ("DataMatrix", True), ("Code93", True)],
)
def test_self_checking_format_detection(fmt, expected):
    """Code 39's check digit is optional and this stock omits it, so a clean
    decode is NOT evidence the payload is right.
    """
    assert is_self_checking(fmt) is expected


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


# ------------------------------------------------- text localizer / ID shapes


def test_id_shape_signature():
    from triagevision.textfind import id_shape

    assert id_shape("EA1568511") == "AADDDDDDD"
    assert id_shape("SN1050837") == "AADDDDDDD"


def test_shape_template_needs_agreement():
    """The template is only trusted when the frame's decoded IDs agree on it."""
    from triagevision.textfind import shape_template

    assert shape_template(["EA1568511", "SN1050837", "EA1568512"]) == "AADDDDDDD"
    assert shape_template(["EA1568511"]) is None          # a single sample proves nothing
    assert shape_template([]) is None


def test_shape_template_rejects_fabricated_ocr_ids():
    """The property that stops OCR inventing patients.

    On a soft frame, OCR readily produces plausible-looking junk that satisfies
    the permissive global ID pattern. Real garbage observed on a sample sheet:
    'SSPGVEPQEB', 'HPT', '28S84A'. None share the batch's shape.
    """
    from triagevision.textfind import id_shape

    template = "AADDDDDDD"
    for junk in ("SSPGVEPQEB", "HPT", "28S84A", "56864"):
        assert id_shape(junk) != template
    for real in ("EA1568511", "SN1050837"):
        assert id_shape(real) == template


@pytest.mark.parametrize(
    "word,w,h,ok",
    [("IMMEDIATE", 674, 100, True),    # a single tag's banner
     ("IMMEDIATE", 1635, 151, False),  # merged across three tags
     ("MINOR", 493, 87, True),
     ("DEAD", 665, 181, True)],
)
def test_word_box_plausibility_rejects_merged_boxes(word, w, h, ok):
    """A box spanning several tags would put the ID band across tags and
    fabricate an ID, so implausible aspect ratios are rejected.
    """
    from triagevision.textfind import _plausible_word_box

    assert _plausible_word_box(word, w, h) is ok


def test_quality_rating_tracks_barcode_success():
    det = _detector()
    img = np.zeros((100, 100, 3), np.uint8)

    def tags_with(decoded, total):
        out = []
        for i in range(total):
            t = _tag(f"EA{i}", i * 10, 0)
            t.barcode = (
                BarcodeRead(f"EA{i}", "Code 39", BBox(0, 0, 5, 5)) if i < decoded else None
            )
            out.append(t)
        return out

    assert det._assess_quality(img, tags_with(10, 10)).rating == "good"
    assert det._assess_quality(img, tags_with(7, 10)).rating == "marginal"
    assert det._assess_quality(img, tags_with(4, 10)).rating == "poor"
    assert det._assess_quality(img, []).rating == "empty"


def test_quality_recommends_retake_only_when_degraded():
    det = _detector()
    img = np.zeros((100, 100, 3), np.uint8)
    good = [_tag("EA1", 0, 0)]
    good[0].barcode = BarcodeRead("EA1", "Code 39", BBox(0, 0, 5, 5))
    q = det._assess_quality(img, good)
    assert q.retake_recommended is False and q.advice is None

    poor = [_tag("EA1", 0, 0), _tag("EA2", 50, 0)]
    q = det._assess_quality(img, poor)
    assert q.retake_recommended is True and "retake" in q.advice.lower()
