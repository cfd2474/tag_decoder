# Handoff: fixing the triage tag decoder server integration

Findings from diagnosing a production audit log against the detector library.
Two independent problems. **The second is the more serious and will not be fixed
by upgrading.**

Library: `https://github.com/cfd2474/tag_decoder`
Required version: **`f8a126a`** or later.

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
