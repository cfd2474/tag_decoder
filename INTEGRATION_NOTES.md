# Handoff: fixing the triage tag decoder server integration

Findings from diagnosing a production audit log against the detector library.
Two independent problems. **The second is the more serious and will not be fixed
by upgrading.**

Library: `https://github.com/cfd2474/tag_decoder`
Required version: **0.6.0** or later.

> **Read "New since 0.3.0" at the bottom first if you have already worked
> through this document once.** The output contract gained several fields,
> including one (`aborted`) that changes how a response must be handled.

---

## Problem 1 — the deployed library predates two bug fixes

The audit log shows both known failure modes.

| Symptom in log | Cause | Fixed in |
|---|---|---|
| `printedTextFound: "BE XEECTANTAY\nSN"` produced no usable acuity | Fuzzy matches were judged on absolute similarity. That string scores 0.74 against EXPECTANT — under the old 0.80 bar — but only 0.35 against every other word, so the category was never ambiguous. Confidence is now the **margin over the runner-up** (≥0.20). | `f8a126a` |
| `printedTextFound: "UNKNOWN"` on two tags (i.e. no banner text read at all) | OCR variants were ordered segmentation-major under a 3-call budget, so every call went to single-line mode and block mode was never reached. Banner bands contain three rows (banner / printed ID / top of barcode), which single-line mode cannot parse. Now preprocessing-major, budget 4. | `da39bc9` |

### Action

```bash
cd /path/to/tag_decoder
git pull
git log --oneline -1     # must show f8a126a or later
python -m pytest tests/  # 84 passed expected
```

### If the deployment is a Docker container, `git pull` on the host does nothing

A rebuild is required, and there are three traps that make a rebuild silently
no-op. All three have to be cleared.

**1. The package version was not bumped until 0.3.0.** Every commit before that
declared `version = "0.1.0"`, so `pip install --upgrade` saw 0.1.0 already
installed and skipped the reinstall — no error, no change. From 0.3.0 onward the
version moves with behaviour, so this stops being a trap. Check the *installed*
version, not the repo:

```bash
docker exec <container> python -c "import triagevision; print(triagevision.__version__)"
```

Must print `0.3.0` or later. `0.1.0` means the old code is still installed.

**2. Docker layer caching.** A line like

```dockerfile
RUN pip install git+https://github.com/cfd2474/tag_decoder.git
```

is byte-identical between builds, so Docker reuses the cached layer and never
re-fetches, no matter what changed upstream. Pin the commit so the instruction
itself changes when the code does:

```dockerfile
RUN pip install --no-cache-dir \
    git+https://github.com/cfd2474/tag_decoder.git@f8a126a
```

**3. The running container is still on the old image.** Rebuilding does not
restart anything:

```bash
docker compose build --no-cache
docker compose up -d --force-recreate
```

### Confirm what is actually loaded, inside the container

```bash
docker exec <container> python -c "
import triagevision, triagevision.ocr as o
print('version     :', triagevision.__version__)      # expect 0.3.0+
print('module path :', triagevision.__file__)
print('MIN_MARGIN  :', getattr(o, 'MIN_MARGIN', 'ABSENT -> OLD'))   # expect 0.2
print('PSM_MODES   :', o.TesseractReader.PSM_MODES)   # expect (7, 6)
"
```

`module path` is the line that usually reveals the problem — it shows which copy
is really being imported. A vendored copy inside the server repo, a bind-mounted
volume, or a stale `__pycache__` will shadow the installed package.

### Proof of whether the upgrade took effect

The OCR path itself changed in `da39bc9`, so the upgraded library produces
*different* `banner_text` for the same image. Same strings as the previous run
means the old code is still executing.

| Tag | Old code emits | 0.3.0 emits | 0.3.0 acuity |
|---|---|---|---|
| SN1050837 | `BE XEECTANTAY\nSN` | `EXPECTANT\nSE` | EXPECTANT |
| EA1568623 | *(no text)* | `TMINOR\nTSE` | MINOR |
| EA1568512 | *(no text)* | `JMMEDIATE\nEP\nLT` | IMMEDIATE |

### If the server is a reimplementation, not this library

The audit log carries fields this library never emits — `finalDecisionSource`,
and colour counts (`reds`, `yellows`, `greens`, `expectants`, `blacks`). If the
server is a port rather than a caller, upgrading changes nothing and the two
fixes have to be ported. Both are small and self-contained:

- **`f8a126a`** — judge a fuzzy keyword match on its **margin over the
  runner-up** (>= 0.20), not on absolute similarity. `XEECTANTAY` scores 0.74
  against EXPECTANT and 0.35 against everything else: unambiguous despite the
  low score. Garbage scores poorly against the whole vocabulary roughly
  equally, and that missing lead is what identifies it.
- **`da39bc9`** — order OCR attempts preprocessing-major (not
  segmentation-major), try page-segmentation modes 7 **and** 6, budget 4 calls.
  Banner bands hold three text rows, which single-line mode cannot parse.

### Verification: replay of the log's own OCR strings under the fixed library

All 15 `printedTextFound` values from the audit log, re-run through the current
matcher. **13 of 13 strings containing real text resolve correctly.** The two
`"UNKNOWN"` entries contained no OCR text at all.

| printedTextFound | Resolves to | score | margin |
|---|---|---|---|
| `DEAD` | DEAD | 1.00 | 1.00 |
| `BE XEECTANTAY\nSN` | **EXPECTANT** | 0.74 | 0.38 |
| `DELAYED\nFAISB` | DELAYED | 1.00 | 1.00 |
| `FIMMEDIATE` | IMMEDIATE | 1.00 | 1.00 |
| `AMINOR E` | MINOR | 1.00 | 1.00 |
| `MMEDIATE` | IMMEDIATE | 0.94 | 0.59 |
| `IMMEDIATE\nNEA` | IMMEDIATE | 1.00 | 1.00 |
| `AINOR` | MINOR | 0.80 | 0.44 |
| `CR\nMMEDIATE\nJ C` | IMMEDIATE | 0.94 | 0.59 |
| `DELAYED\nEAISS` | DELAYED | 1.00 | 1.00 |

Note `AINOR` → MINOR at 0.80 and `BE XEECTANTAY\nSN` → EXPECTANT at 0.74. Both
would be rejected by an absolute-similarity threshold and both are correct.
**Do not reintroduce a minimum-score gate in the integration layer.**

### Not the cause: image downscaling

The server processes at 2400x1800 (downscaled from 4128x3096). Tested both
sizes: 15/15 tags with zero UNKNOWN acuities at each. Downscaling is fine and
is ~30% faster. Leave it.

---

## Problem 2 — the integration silently converts UNKNOWN into a real category

**This is the priority fix.** It is in the server/mapping layer, not the library.

Three tags in the log had no readable acuity. All three were emitted as
`"category": "Immediate"`, formatted identically to the twelve tags that were
read cleanly:

| Tag | printedTextFound | Emitted | Ground truth |
|---|---|---|---|
| SN1050837 | `BE XEECTANTAY\nSN` | Immediate | **EXPECTANT** |
| EA1568623 | `UNKNOWN` | Immediate | **MINOR** |
| EA1568512 | `UNKNOWN` | Immediate | IMMEDIATE *(coincidentally right)* |

**EA1568623 is a MINOR tag — walking wounded — reported as Immediate.** The
library returned `UNKNOWN` for these, meaning "could not read, needs a human."
Something downstream is turning that into a definite category with no marker.

Defaulting an unreadable tag to the most urgent category is a defensible
clinical choice (over-triage beats under-triage). Doing it *silently* is not:
an operator cannot tell a read tag from an unread one, so the failure never
gets corrected.

### Required changes

1. **Never map `UNKNOWN` to a triage category without marking it.** Either pass
   `UNKNOWN` through, or if a fail-safe default is wanted, emit the default
   *plus* an explicit flag (e.g. `"assumed": true`, `"reason": "acuity
   unreadable"`) so the UI can distinguish it.
2. **Propagate `confidence`.** The library emits it per tag; the audit log drops
   it. Those three tags scored ~0.45, well below the 0.60 review line. Surface
   it, and treat `< 0.60` as needing human review.
3. **Propagate `warnings`.** Per-tag strings state exactly what went wrong
   (`banner text unreadable`, `treat the id as unverified`, `verify this tag`).
   The log drops these entirely. They are the safety channel — do not filter
   them at the boundary.

---

## Problem 3 — check the DEAD / MORGUE mapping

The log maps `DEAD` → `"Deceased"`. No MORGUE tag appears in this image so the
behaviour is unverified, but **if both collapse to `"Deceased"` that is a bug.**

`DEAD` and `MORGUE` are the same clinical outcome under two different triage
protocols. They are deliberately separate values because the consuming system
needs to know which tag was physically seen. Keep them distinct in the category
field. For logic that genuinely should treat them alike, the library exposes
`acuity.is_deceased`, which is true for both.

---

## Contract the integration must preserve

### Acuity values

`IMMEDIATE`, `DELAYED`, `MINOR`, `EXPECTANT`, `DEAD`, `MORGUE`, `UNKNOWN`.

If mapping to different labels, the mapping must be 1:1. Do not merge, and do
not substitute a category for `UNKNOWN` without flagging.

### Duplicate patient IDs must survive

One line item per **physical tag**. The same patient ID on two tags produces two
entries, whether or not the acuities agree. This is by design — reconciling
duplicates is the consuming system's job. The library already de-duplicates
repeat detections of a *single* tag (same ID + overlapping geometry), so any
remaining repeat is a real second tag.

Verified: a frame with SN1050837 on both an EXPECTANT and a DEAD tag returns two
entries, no warning. The current log correctly shows 15 entries from 10 unique
IDs — this part is working.

### Per-tag fields the integration should not drop

| Field | Why it matters |
|---|---|
| `patient_id` | may be `null` if the tag was located but not identified |
| `acuity` | may be `UNKNOWN` — see Problem 2 |
| `confidence` | 0–1; **below 0.60 means a human should look** |
| `warnings` | per-tag failure detail; the safety channel |
| `banner_text` | **raw OCR only** (`"SDELAYED"`, `"CR\nMMEDIATE\nJ C"`). Diagnostic. Never parse this — `acuity` is the parsed result |
| `barcode.format` | e.g. `"Code 39"` (note the space) |

Top level: `count` (line items) and `identified_count` (those with an ID).

### Confidence composition

`0.40` ID recovered · `0.55` acuity from banner text · `+0.05` barcode matches
the printed ID exactly · `−0.20` barcode contradicts it.

---

## Known limitation to design around, not fix in software

The tag stock uses **Code 39 with no check digit**, so a misread can decode
cleanly to a wrong patient ID. Patient IDs are pre-printed vendor serials in an
unknown format, so the ID pattern must stay permissive and cannot reject a bad
decode on shape.

The only guard is the printed ID line, cross-checked against the barcode payload
— and it requires an **exact** match, because serials are sequential and a
one-digit misread scores 0.89 similarity. Tags where it could not be confirmed
carry `treat the id as unverified`. On dense frames roughly half the IDs land in
that state.

Do not filter that warning. Code 128 stock would eliminate this class of risk
outright; that is a procurement decision, not a code change.

---

## Acceptance criteria

Re-run the same image after upgrading. Expect:

- 15 line items, 15 identified
- `SN1050837` → **EXPECTANT** (not Immediate)
- `EA1568623` → **MINOR** (not Immediate)
- Zero `UNKNOWN` acuities
- Any tag that *is* `UNKNOWN` in future is visibly distinguishable in the output
- `confidence` and `warnings` present on every tag

---

## New since 0.3.0 — additional contract the integration must handle

Four releases added capability aimed squarely at poor field photographs. Each
adds output the integration has to deal with; the first is not optional.

### 1. A response can now be ABORTED — handle this first

A readability check runs concurrently with decoding and stops a frame that is
not worth processing. When it fires:

```json
{ "tags": [], "count": 0, "aborted": true,
  "preflight": {"rating": "degraded", "symbols_found": 0, "words_found": 8, ...},
  "image_quality": {"rating": "poor", "retake_recommended": true, "advice": "..."},
  "warnings": ["image quality poor: ..."] }
```

**This is the retake trigger.** `aborted: true` means processing stopped
deliberately and the operator should be asked for a new photo. It is not an
error and not an empty scene — those are distinguishable via
`preflight.rating`:

| `preflight.rating` | Meaning |
|---|---|
| `ok` | a barcode decoded on the fast path; frame processed normally |
| `degraded` | nothing decoded but banner words present; IDs would be OCR-only at best |
| `unusable` | neither symbols nor words; nothing here to read |

Aborting takes ~1.5 s instead of ~7 s spent producing a poor answer, and a
cleaner retake is also a faster one.

An integration that ignores `aborted` entirely still behaves safely — it sees
zero tags rather than bad data — but it will not know to prompt for a retake,
which is the whole point.

Policy is a library-side config (`preflight_abort_on`): `"degraded"` (default),
`"unusable"`, or `"never"`. The default trades data for a retake prompt: a frame
that would have yielded acuities with OCR-derived IDs returns nothing instead.
That is right when a better photo can be taken and **wrong if the scene is
gone** — `"never"` keeps whatever can be read while still reporting the verdict.

### 2. `image_quality` on every non-aborted response

```json
"image_quality": {"rating": "good", "retake_recommended": false,
                  "barcode_decode_rate": 1.0, "id_verified_rate": 0.87,
                  "tags_found": 7, "sharpness": 617.7, "advice": null}
```

`retake_recommended` is the flag to act on; `advice` is operator-facing text,
safe to display verbatim. A frame can process successfully and still be rated
`marginal` or `poor` — surface that.

Note `sharpness` is reported but is **not** the basis of the rating and must not
be used as one. Sharpness does not predict readability here: one sheet read
15/15 after being blurred well below the sharpness of another that read 7/15.
The rating is based on outcomes — how many located tags gave up a decodable
barcode.

### 3. Two new per-tag fields — `id_source` and `found_by`

Tags can now be located by their printed banner word when the barcode fails to
decode, with the patient ID read off the printed line. On a soft sheet this
took results from 7 tags to 15.

| Field | Values | Why it matters |
|---|---|---|
| `id_source` | `"barcode"` \| `"ocr"` \| `null` | `"ocr"` means the ID was **read as text and may contain character errors** (0/O, 1/I, 5/S, 8/B). `null` means no ID was recovered — the tag still has a valid acuity. |
| `found_by` | `"barcode"` \| `"text"` \| `"color"` | How the tag was located. |

**Do not treat an `ocr` ID as equivalent to a `barcode` one.** If the consuming
system matches patients by ID, an OCR-derived ID should be treated as a
candidate needing confirmation, not a key.

A tag with `patient_id: null` but a valid `acuity` is a real, deliberate
outcome, not a bug: the tag was found and its category read, but no ID could be
recovered to a form matching the other tags in the frame. Reporting the acuity
without an ID beats inventing a patient number.

### 4. Fabricated IDs are actively suppressed

OCR readily produces plausible-looking junk that satisfies the permissive ID
pattern — real examples from a sample sheet: `SSPGVEPQEB`, `HPT`, `28S84A`.
Because tags in one photo come from one batch, any barcode that decodes reveals
that batch's ID shape, and OCR candidates must match it. That took fabricated
IDs from six to zero on the frame in question.

The consequence for the integration: **more `patient_id: null` entries, and
fewer wrong ones.** That is the intended trade.

### Updated verification snippet

```bash
docker exec <container> python -c "
import triagevision, triagevision.ocr as o
print('version     :', triagevision.__version__)      # expect 0.6.0+
print('module path :', triagevision.__file__)
print('MIN_MARGIN  :', getattr(o, 'MIN_MARGIN', 'ABSENT -> OLD'))
print('preflight   :', __import__('triagevision.preflight', fromlist=['x']).PROBE_ANGLE)
"
```

`preflight` importing at all proves the build is 0.5.0 or later.

### Updated acceptance criteria

Re-run the sheet with mixed IDs and orientations. Expect:

- 15 line items, `SN1050837` → EXPECTANT, `EA1568623` → MINOR, zero `UNKNOWN`
- `confidence`, `warnings`, `id_source`, `found_by` present on every tag
- `image_quality` present, `rating: "good"`, `retake_recommended: false`
- `aborted: false`

Then re-run a deliberately soft photo. Expect `aborted: true`,
`preflight.rating: "degraded"`, empty `tags`, and an `advice` string suitable
for prompting a retake.
