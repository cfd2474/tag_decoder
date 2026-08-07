# triagevision

Reads a photograph containing any number of triage tags and returns, per tag,
the **patient ID** and the **acuity level** (IMMEDIATE / DELAYED / MINOR /
EXPECTANT / DEAD).

Designed to run as a module inside an existing server — the detector is a plain
Python class with no web framework dependency. An optional FastAPI wrapper and a
CLI are included.

```python
from triagevision import TriageTagDetector

detector = TriageTagDetector()          # build once at process start
result = detector.detect("scene.jpg")   # path, raw bytes, or a numpy BGR array

result.roster()
# [{'patient_id': 'EA1568511', 'acuity': 'IMMEDIATE'},
#  {'patient_id': 'EA1568519', 'acuity': 'MINOR'}, ...]
```

`TriageTagDetector` is stateless and thread-safe. Build it once and share it.

---

## How it decides

Three signals, ordered by how much they can be trusted:

| Signal | Used for | Why it ranks there |
|---|---|---|
| **Banner text** | acuity | Invariant to lighting. Closed five-word vocabulary, so fuzzy matching is safe — a mangled read still lands on the right category. |
| **Barcode** | patient ID | Open vocabulary, so it must be exact; a symbol either decodes or it does not. |
| **Printed ID line** | cross-check | Catches a silent barcode misread, and stands in when the symbol is glare-blown. |
| **Field colour** | *off by default* | Lighting-dependent. See below. |

The printed word is the primary determinant of acuity, not the colour. A red tag
under sodium-vapour light measures orange; a green one under a warm lamp
measures yellow — a one-category error in the dangerous direction. The word
IMMEDIATE is still the word IMMEDIATE.

### Colour is off by default

Measured on a nine-tag photo under simulated lighting (same frame, colour cast
applied). "acuity" counts tags given the correct category:

| Scene | acuity, colour **off** | acuity, colour **on** |
|---|---|---|
| baseline | 9/9 | 9/9 |
| sodium vapour | 9/9 | 9/9 |
| blue LED | **9/9** | **0/9** |
| dim / dusk | 9/9 | 9/9 |
| overexposed | 7/9 | 7/9 |
| green work lamp | 7/9 | 8/9 |
| **total** | **50/54 in 39s** | 51/54 in 66s |

Colour does not merely fail to help under a colour cast — it does active harm. A
white barcode label under a blue scene light segments *as a blue tag*, so the
inferred tag box collapses onto the label and the banner falls outside the crop
entirely. That is the 0/9.

Enable it with `DetectorConfig(use_color=True)` only if you need acuity for tags
whose barcode cannot be decoded at all; it is the sole localizer that works
without a symbol.

### Rotation

Tags photographed on the ground land at arbitrary angles, and each tag's
banner / printed ID / barcode stack rotates *with that tag*, independently of its
neighbours. Orientation is therefore recovered per tag, from the tag itself.

The mechanism is the barcode's **reading direction**. Decoders report a 1D
symbol's corners in reading order, which is a full 360° orientation — unlike a
minimum-area rectangle, which is only defined up to 90° and cannot tell a tag
from the same tag upside down. Aligning the bars left-to-right puts the banner on
top by construction, at any angle.

| Frame rotation | acuity | IDs |
|---|---|---|
| 0° / 90° / 180° | 9/9 | 9/9 |
| 270° | 8/9 | 9/9 |
| 37° / −23° | 8/9 | 8/9 |

Arbitrary angles need the rotation sweep (below), since zxing scans along rows
and has blind bands between its 90° retries — an unswept frame rotated 37° drops
from nine decoded symbols to **zero**.

---

## Output

```json
{
  "tags": [{
    "patient_id": "EA1568511",
    "acuity": "IMMEDIATE",
    "confidence": 1.0,
    "bbox": [903, 1832, 1104, 505],
    "barcode": {"text": "EA1568511", "format": "Code39", "bbox": [...], "quad": [...]},
    "banner_text": "IMMEDIATE",
    "color": null,
    "warnings": []
  }],
  "count": 1,
  "image_size": {"width": 3000, "height": 4000},
  "elapsed_ms": 5412.7,
  "warnings": []
}
```

`confidence` is interpretable, not a model score:

- `0.40` patient ID recovered
- `0.55` acuity from banner text (full on an exact word match, scaled on a fuzzy one)
- `±0.05 / −0.20` barcode agrees / disagrees with the printed ID line

**Treat anything below 0.60 as needing a human look.** A tag whose acuity came
from colour alone tops out near 0.57 by design — that is the honest position when
the only evidence is lighting-dependent.

`warnings` is the channel for everything a dispatcher needs to know: an
unreadable banner, a barcode that disagrees with the printed ID, the same patient
ID on two tags. Do not discard it.

### Safety behaviour

- OCR noise never invents a category. Fuzzy matching is length-guarded and
  floored at 0.72 similarity; a weak match with nothing corroborating it returns
  `UNKNOWN`. A confidently wrong triage category is worse than an unread one.
- Most vendors print DEAD and EXPECTANT on the same black field, so colour cannot
  separate them. With no readable banner, that returns `UNKNOWN`, not a coin flip.
- `DetectorConfig(require_text=True)` refuses to infer acuity from colour at all.

---

## Install

```bash
pip install -r requirements.txt
```

The tesseract **binary** is a system package and is installed separately:

```bash
brew install tesseract          # macOS
apt-get install tesseract-ocr   # Debian/Ubuntu
```

Without it the detector still returns patient IDs, but acuity falls back to the
colour path and every result carries a warning saying so. Check
`detector.reader.available` at startup.

## CLI

```bash
python -m triagevision.cli scene.jpg --roster
```

```bash
python -m triagevision.cli *.jpg --annotate ./out --workers 4
```

## HTTP service

```bash
pip install fastapi "uvicorn[standard]" python-multipart
uvicorn triagevision.service:app --host 0.0.0.0 --port 8080
```

`POST /detect` (multipart field `image`, optional `?roster_only=true`) and
`GET /healthz`, which reports whether OCR is actually available.

---

## Performance

~5–7 s for a nine-tag 12 MP phone photo on a laptop CPU; ~10 s when the rotation
sweep runs in full. Tags within a frame are read in parallel (`max_workers`).

Cost is dominated by OCR, because `pytesseract` shells out to the tesseract
binary once per call. If you need more throughput, in order of payoff:

1. **Downscale before submitting.** 12 MP is far more than needed; the barcode
   stage is the only part wanting the resolution.
2. **Swap the OCR backend.** `pytesseract`'s per-call process spawn is the
   bottleneck, not the recognition. Implement the `TextReader` protocol
   (`read`, `read_variants`, `read_id`) against `tesserocr` (in-process API),
   PaddleOCR, or a GPU service and pass it to the constructor:
   ```python
   TriageTagDetector(text_reader=MyReader())
   ```
3. **Set `max_workers=1`** if you already run several detections concurrently, so
   the two levels of parallelism don't oversubscribe the box.

## Configuration

Everything vendor- or deployment-specific is in `DetectorConfig`, loadable from
JSON via `DetectorConfig.from_json(path)`. The values most likely to need
changing for a different tag stock:

| Field | Default | Meaning |
|---|---|---|
| `patient_id_pattern` | permissive alphanumeric | Tighten to your own ID scheme — it is what rejects a garbage decode. |
| `text_keywords` | START/SALT vocabulary | Add local wording (e.g. `MINIMAL`, `WALKING WOUNDED`). |
| `tag_width_per_barcode_width` | `1.45` | Tag width as a multiple of symbol width. |
| `tag_aspect_prior` | `2.25` | Tag width ÷ height. |
| `barcode_offset_frac` | `0.17` | How far the symbol sits below tag centre. |
| `require_text` | `False` | Never infer acuity from colour. |
| `use_color` | `False` | Enable colour segmentation. |

The three geometry priors are measured from standard START-scheme tags. To
re-measure for different stock, decode one sheet and compare each symbol's width
and centre against the tag outline — note that **symbol height must not be used**:
across one sample sheet the same tag design reported heights from 6 px to 134 px
while widths held within 2%.

## Tests

```bash
python -m pytest tests/
```

The suite is unit-level and needs no fixture images. To run the end-to-end check
against a real sheet:

```bash
TRIAGEVISION_SAMPLE=photo.jpg \
TRIAGEVISION_TRUTH='{"EA1568511":"IMMEDIATE"}' \
python -m pytest tests/
```

## Limits

- **Overexposed frames are the weak case** (7/9). Blown highlights destroy the
  bars and the lettering together, and no amount of downstream logic recovers
  information that is not in the pixels. Expose for the tags.
- **A tag with no decodable barcode and no colour is invisible.** With
  `use_color=False` the barcode is the only localizer. If torn or obscured tags
  matter, enable colour and accept the trade-offs above.
- **Code 39 without a check digit can decode to the wrong string.** The printed-ID
  cross-check is what catches this; do not disable OCR and then trust IDs blindly.
- Not a medical device. Output is decision *support* — the confidence score and
  warnings exist so a human stays in the loop.

## License

MIT — see [LICENSE](LICENSE).
