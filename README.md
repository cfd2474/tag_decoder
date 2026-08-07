# triagevision

Reads a photograph containing any number of triage tags and returns, per tag,
the **patient ID** and the **acuity level**.

Categories are the exact words printed on the tag: `IMMEDIATE`, `DELAYED`,
`MINOR`, `EXPECTANT`, `DEAD`, `MORGUE`, plus `UNKNOWN` when the banner cannot be
read confidently.

**`DEAD` and `MORGUE` are deliberately not merged.** They are the same clinical
outcome under two different triage protocols, and collapsing them would destroy
information the consuming system needs. Callers that genuinely want to treat them
alike should say so explicitly via `acuity.is_deceased`.

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
| 37° | 8/9 | 8/9 |
| −23° | 7/9 | 8/9 |

Arbitrary angles need the **rotation sweep**: zxing scans along rows and retries
at 90° steps, leaving blind bands in between. An unswept frame rotated 37° drops
from nine decoded symbols to **zero**.

The sweep always runs to completion, at half scale (~0.6 s). It deliberately has
no early exit — there is no way to know how many tags a frame contains, so
"we already found some, stop looking" silently drops patients. Measured on a
five-tag frame: the cheap passes found two, an early exit stopped at 30°, and
the other three only decoded at 60° and 75°.

A second five-tag photo, tags at four different orientations including 90° and
inverted, plus a MORGUE tag: **5/5 in 3.2 s**.

### Crowded frames need tiled decoding

A whole-frame scan does not reliably return every symbol present. On a
fifteen-tag sheet, one pass returned two instances of some repeated payloads but
only one of others — and with colour off the barcode is the sole localizer, so a
skipped symbol is a patient who silently disappears. Both symbols it missed
decoded immediately once cropped.

So every frame is also re-scanned in overlapping tiles, at **two scales**. One
grid is not enough, and a "best" grid does not exist: every single grid tried
missed at least one symbol, and *which* one changed with the grid, because the
decoder's response depends on how a symbol happens to sit inside its scan
window. Coarse tiles catch symbols needing surrounding context; fine tiles catch
those that only decode cropped close. Together: 15/15. Separately: never more
than 14.

One symbol needed a third condition — a tight crop **and** glare suppression
applied at tile scale. Full-frame illumination correction did not rescue it,
because a specular highlight is local and flattening it across 12 MP leaves the
local gradient intact. Tiles are therefore scanned native *and* glare-flattened.

Tiles decode in parallel (zxing releases the GIL), which is what keeps this
affordable.

### Category coverage

All six printed categories have been read from real tags:

| Category | Field | Verified |
|---|---|---|
| IMMEDIATE | red | ✓ |
| DELAYED | yellow | ✓ |
| MINOR | green | ✓ |
| EXPECTANT | slate blue | ✓ |
| DEAD | black | ✓ (exact read; fuzzy disabled) |
| MORGUE | black | ✓ |

`EXPECTANT` is commonly described as a grey tag but is printed slate blue on the
stock measured here (H=107 S=82 V=137), so the colour band covers both. `DEAD`
and `MORGUE` share the black field and are separated only by the word.

---

## Output reference

`detect()` returns a `DetectionResult`. `.to_dict()` gives the JSON below; it is
the same shape the CLI and the HTTP endpoint emit.

```json
{
  "tags": [
    {
      "patient_id": "EA1568519",
      "acuity": "DELAYED",
      "confidence": 0.95,
      "bbox": [1002, 672, 787, 447],
      "color": null,
      "barcode": {
        "text": "EA1568519",
        "format": "Code 39",
        "bbox": [1139, 943, 534, 8],
        "quad": [[1139, 943], [1672, 943], [1673, 951], [1140, 951]]
      },
      "banner_text": "SDELAYED",
      "warnings": [
        "patient id came from a single Code 39 decode with no check character, and the printed id line could not be read to confirm it; treat the id as unverified"
      ]
    }
  ],
  "count": 7,
  "identified_count": 7,
  "image_size": {"width": 3000, "height": 4000},
  "elapsed_ms": 3187.4,
  "warnings": []
}
```

### Top level

| Field | Type | Description |
|---|---|---|
| `tags` | array | One entry per physical tag read. Ordered top-to-bottom, then left-to-right, by bounding-box centre. |
| `count` | int | Number of line items in `tags`. Two tags sharing a patient ID count as two. Also `result.tag_count`. |
| `identified_count` | int | How many of those yielded a patient ID. Lower than `count` only when a tag was located but its ID could not be recovered. Also `result.identified_count`. |
| `image_size` | object | `{"width": int, "height": int}` of the submitted image, in pixels. |
| `elapsed_ms` | float | Wall-clock processing time for this image. |
| `warnings` | string[] | **Frame-level** notices, not tag-specific — currently only the missing-OCR-backend notice. Empty on a healthy run. Per-tag problems live on each tag. |

### Per tag — `tags[]`

| Field | Type | Description |
|---|---|---|
| `patient_id` | string \| `null` | The patient ID. `null` when a tag was located but no ID could be recovered — check `warnings` for why. |
| `acuity` | string | One of `IMMEDIATE`, `DELAYED`, `MINOR`, `EXPECTANT`, `DEAD`, `MORGUE`, `UNKNOWN`. `DEAD` and `MORGUE` are never merged. |
| `confidence` | float 0–1 | Interpretable score, composed below. **Below 0.60 warrants a human look.** |
| `bbox` | `[x, y, w, h]` | Axis-aligned box in image pixels. This is the *upright* box around a possibly rotated tag, so it is larger than the tag itself when tilted. Use `barcode.quad` for true orientation. |
| `color` | object \| `null` | `null` by default, since colour segmentation is off. Populated only with `use_color=True`. |
| `barcode` | object \| `null` | `null` when no symbol decoded — in that case the ID, if any, came from OCR of the printed line and `warnings` will say so. |
| `banner_text` | string \| `null` | **Raw, unparsed OCR output** of the banner, e.g. `"SDELAYED"`, `"EDEADY"`, `"V MINOR"`. Diagnostic only. `acuity` is the parsed result — do not parse this field yourself. |
| `warnings` | string[] | Per-tag caveats. See below. |

### `barcode`

| Field | Type | Description |
|---|---|---|
| `text` | string | Raw decoded payload, before ID-pattern validation. Usually equals `patient_id`; differs if the payload failed validation. |
| `format` | string | Symbology as the decoder reports it, e.g. `"Code 39"`, `"Code 128"` — note the space. Determines whether a clean decode is self-verifying. |
| `bbox` | `[x, y, w, h]` | Symbol bounds in image pixels. **Height is unreliable** — decoders report it erratically (6 px to 134 px across identical tags). Width is trustworthy. |
| `quad` | `[[x, y] × 4]` | Corners in **reading order**, so `quad[0] → quad[1]` is the symbol's reading direction and gives the tag's full 360° orientation. |

### `color` — only when `use_color=True`

| Field | Type | Description |
|---|---|---|
| `name` | string | `red`, `yellow`, `green`, `slate`, `black`. |
| `acuity_candidates` | string[] | Categories this field colour permits. `black` gives `["DEAD", "MORGUE"]` — colour alone cannot separate those two. |
| `score` | float 0–1 | Separation from the next-best colour. Near 1.0 is a clean single-colour field. |
| `coverage` | float 0–1 | Fraction of the region matching this band. |

### How `confidence` is composed

| Weight | Condition |
|---|---|
| `0.40` | patient ID recovered |
| `0.55` | acuity from banner text — full on an exact word match, scaled by similarity on a fuzzy one |
| `+0.05` | barcode matches the printed ID line exactly |
| `−0.20` | barcode clearly contradicts the printed ID line |

**Treat anything below 0.60 as needing a human look.** A tag whose acuity came
from colour alone tops out near 0.57 by design — the honest position when the
only evidence is lighting-dependent.

### Per-tag `warnings` you will actually see

| Message (abbreviated) | Meaning |
|---|---|
| `…treat the id as unverified` | The ID rests on a single Code 39 decode with no check character, and the printed line could not be read to confirm it. Not an error — just uncorroborated. |
| `barcode says X but the printed id reads Y … verify this tag` | The two disagree substantially (similarity < 0.50). The barcode value is kept. |
| `banner text unreadable` / `matched no known category` | The acuity word could not be read or did not match the vocabulary. |
| `acuity inferred from field color … verify before acting` | Banner unreadable, colour used as fallback. Only possible with `use_color=True`. |
| `no readable barcode and no readable printed id` | Tag located but unidentifiable; `patient_id` is `null`. |

Do not filter these out of your downstream feed — with a check-digit-less
symbology they are the only signal distinguishing a corroborated ID from an
uncorroborated one.

### Python accessors

| Expression | Returns |
|---|---|
| `result.tag_count` | `int` — line items, same as `count` |
| `result.identified_count` | `int` — those with a patient ID |
| `result.roster()` | `[{"patient_id": str, "acuity": str}]` — minimal payload, repeats preserved |
| `result.tags[i].acuity.is_deceased` | `bool` — `True` for both `DEAD` and `MORGUE`, for callers that need to treat them alike |
| `annotate(image, result)` | BGR image with boxes and labels drawn, for debugging |

### Duplicate patient IDs are reported, not reconciled

One entry per physical tag, always. If the same patient ID appears on two tags,
both are returned as separate line items — exactly as they would be with
different IDs, and **whether or not their acuities agree**:

```python
detector.detect("seven_tags.jpg").roster()
# [{'patient_id': 'EA1568519', 'acuity': 'DELAYED'},
#  {'patient_id': 'SN1050837', 'acuity': 'EXPECTANT'},
#  {'patient_id': 'EA1568511', 'acuity': 'MORGUE'},
#  ...
#  {'patient_id': 'SN1050837', 'acuity': 'DEAD'}]
```

No warning is raised and nothing is collapsed. Deciding what a duplicate means —
a re-tag, a misread, a vendor batch collision — belongs to the consuming system,
which has context this one does not. Hiding a tag here would conceal something
that physically exists.

The one thing that *is* de-duplicated is a single physical tag detected twice
(same ID **and** overlapping geometry), since counting one piece of card as two
patients would corrupt `count`.

### Safety behaviour

- OCR noise never invents a category. Fuzzy matching is length-guarded, and the
  similarity floor scales with word length. A confidently wrong triage category
  is worse than an unread one.
- **A match is judged on its lead over the runner-up, not its raw score.**
  Measured across real reads and real OCR garbage, the two separate completely
  on margin and only partially on absolute similarity:

  | | best score | margin over runner-up |
  |---|---|---|
  | real banner reads | 0.696 – 0.933 | **0.267 – 0.600** |
  | OCR garbage | 0.190 – 0.444 | **0.000 – 0.111** |

  `XEECTANTAY` scores just 0.74 against EXPECTANT — OCR chewed both ends — but
  0.35 against everything else, so the category is not in doubt. Garbage scores
  poorly against the whole vocabulary roughly equally, and that missing lead is
  what identifies it. A match with a margin below 0.20 and nothing corroborating
  it returns `UNKNOWN`.
- **`DEAD` requires an exact read.** At four characters, a genuine OCR slip
  (`DEAO`, `OEAD`) scores 0.750 against `DEAD` — and so do `READ`, `HEAD`,
  `BEAD` and `DEED`. No threshold admits the slips while rejecting the real
  words, so fuzzy matching is disabled for it entirely. `MINOR` (5 chars, slip
  0.800 vs nearest confusable `MAJOR` 0.600) uses 0.78; 6+ characters use 0.72.
- `DEAD` and `MORGUE` print on the same black field, so colour cannot separate
  them — only the word can. With no readable banner that returns `UNKNOWN`,
  not a coin flip between two protocols. (`EXPECTANT` is a grey field.)
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

Measured on 12 MP phone photos, laptop CPU, all correct:

| Frame | Tags | Time |
|---|---|---|
| single tag | 1 | 2.5 s |
| five tags | 5 | 3.8 s |
| nine tags | 9 | 4.5 s |
| fifteen tags | 15 | 5.9 s |
| sixteen tags | 16 | 4.7 s |

Roughly 2.5 s of fixed cost (whole-frame decode, glare pass, two tile scales,
rotation sweep) plus ~0.2 s per tag. Both the tile decodes and the per-tag
reading run in parallel (`max_workers`).

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
| `patient_id_pattern` | permissive alphanumeric | Keep it loose unless you genuinely control the ID format — see below. |
| `text_keywords` | START/SALT vocabulary | Add local wording (e.g. `MINIMAL`, `WALKING WOUNDED`). |
| `tag_width_per_barcode_width` | `1.45` | Tag width as a multiple of symbol width. |
| `tag_aspect_prior` | `2.25` | Tag width ÷ height. |
| `barcode_offset_frac` | `0.17` | How far the symbol sits below tag centre. |
| `require_text` | `False` | Never infer acuity from colour. |
| `use_color` | `False` | Enable colour segmentation. |

**On `patient_id_pattern`:** it is tempting to tighten this to your observed IDs,
and it is usually wrong. Patient numbers are pre-printed vendor serials, so the
format of the next batch is not knowable — a pattern fitted to today's stock
silently drops every patient in a batch that looks different, and it fails
closed in the worst way: quietly, on real casualties.

That has a consequence. With a permissive pattern, shape cannot reject a bad
decode, and Code 39 — the symbology on this stock — has only an *optional* check
digit, which these tags do not carry. So a misread can decode cleanly to a wrong
string with nothing structural to catch it. The **printed ID line is the only
real guard**.

It is therefore treated strictly. Confirmation requires the printed line to match
the payload **exactly**, not merely closely: serials are sequential, so the
likeliest misread is one digit — and `EA1568513` against `EA1568512` scores 0.89
similarity. Any threshold loose enough to tolerate OCR noise would bless exactly
the error the check exists to catch. Three outcomes:

| Printed line | Meaning | Effect |
|---|---|---|
| exact match | confirmed | `+0.05` confidence |
| close but not identical | OCR noise, not evidence either way | flagged `treat the id as unverified` |
| clearly different (<0.50) | possible misread, or a neighbour's line in the crop | flagged `verify this tag`, `−0.20` |

Unreadable printed line, on a symbology with no check character, also flags
`treat the id as unverified`. Do not disable OCR and then trust IDs blindly, and
do not filter these warnings out of your downstream feed.

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
