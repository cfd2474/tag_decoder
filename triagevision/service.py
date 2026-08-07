"""Optional HTTP wrapper: `uvicorn triagevision.service:app`.

The detector itself has no web dependency -- import `TriageTagDetector` directly
if you are embedding it in an existing service. This module exists so the thing
can be run as a standalone sidecar without writing any glue.

Install the extra first:  pip install "fastapi" "uvicorn[standard]" "python-multipart"
"""

from __future__ import annotations

import logging
import os

try:
    from fastapi import FastAPI, File, HTTPException, Query, UploadFile
    from fastapi.responses import JSONResponse
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        'the HTTP service needs extra packages: pip install fastapi "uvicorn[standard]" '
        "python-multipart"
    ) from exc

from . import __version__
from .config import DetectorConfig
from .detector import TriageTagDetector

log = logging.getLogger(__name__)

# Reject oversized uploads before decoding them: a 12MP phone photo is ~4MB, so
# this leaves generous headroom while bounding memory per request.
MAX_UPLOAD_BYTES = int(os.getenv("TRIAGEVISION_MAX_UPLOAD_BYTES", 24 * 1024 * 1024))


def build_detector() -> TriageTagDetector:
    cfg = DetectorConfig()
    path = os.getenv("TRIAGEVISION_CONFIG")
    if path:
        cfg = DetectorConfig.from_json(path)
    # One shared reader thread pool per request would oversubscribe a busy
    # server, so let the deployment decide.
    if os.getenv("TRIAGEVISION_WORKERS"):
        cfg.max_workers = int(os.environ["TRIAGEVISION_WORKERS"])
    return TriageTagDetector(cfg)


app = FastAPI(title="triagevision", version=__version__)
detector = build_detector()


@app.get("/healthz")
def healthz() -> dict:
    """Liveness plus the two facts that actually determine output quality."""
    return {
        "status": "ok",
        "version": __version__,
        "ocr_backend_available": detector.reader.available,
        "color_segmentation": detector.cfg.use_color,
    }


@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    roster_only: bool = Query(
        False, description="return only patient_id/acuity pairs"
    ),
) -> JSONResponse:
    """Detect every triage tag in one uploaded image."""
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"image exceeds {MAX_UPLOAD_BYTES} bytes",
        )

    try:
        result = detector.detect(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        log.exception("detection failed for %s", image.filename)
        raise HTTPException(status_code=500, detail="detection failed") from None

    if roster_only:
        return JSONResponse({"tags": result.roster(), "count": result.tag_count})
    return JSONResponse(result.to_dict())
