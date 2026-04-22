# ============================================================
# BG Remover — Server
# FastAPI server that runs RMBG-1.4 for background removal.
#
# Endpoints:
#   GET  /health           → {"status": "ok", "device": "cuda"|"cpu"}
#   POST /remove-bg/image  → multipart "file"  →  PNG bytes (RGBA)
#
# This server is designed to be called by bg_remover_addon.py
# when the addon is set to SERVER mode.
#
# Run:
#   pip install -r requirements.txt
#   python bg_remover_server.py
#   # or: uvicorn bg_remover_server:app --host 0.0.0.0 --port 8000
# ============================================================

import io
import logging
import os
from contextlib import asynccontextmanager

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("bg_remover_server")

# ── Config ────────────────────────────────────────────────────────────────────

HOST = os.environ.get("BG_REMOVER_HOST", "0.0.0.0")
PORT = int(os.environ.get("BG_REMOVER_PORT", "8000"))
MODEL_ID = os.environ.get("BG_REMOVER_MODEL", "briaai/RMBG-1.4")
# Upload size cap (50 MB default) to stop someone from OOMing the server
MAX_UPLOAD_MB = int(os.environ.get("BG_REMOVER_MAX_UPLOAD_MB", "50"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# ── Model state ───────────────────────────────────────────────────────────────

_model = None
_processor = None
_device = "cpu"


def _load_model():
    """Load RMBG-1.4 once at startup. Keeps it hot for every request."""
    global _model, _processor, _device

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading {MODEL_ID} on {_device.upper()}…")

    _model = AutoModelForImageSegmentation.from_pretrained(
        MODEL_ID, trust_remote_code=True
    )
    _model.to(_device).eval()

    _processor = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
    ])
    log.info("Model ready.")


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield
    # no teardown needed; process exit releases GPU mem


app = FastAPI(
    title="BG Remover Server",
    version="1.0.0",
    description="RMBG-1.4 background removal server for bg_remover_addon.py",
    lifespan=lifespan,
)

# Permissive CORS so a local web UI can hit this too. Tighten if exposing publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Core ──────────────────────────────────────────────────────────────────────

def _remove_bg(pil_img: Image.Image) -> Image.Image:
    """Run RMBG-1.4 on a PIL image, return RGBA with alpha = predicted mask."""
    W, H = pil_img.size
    rgb = pil_img.convert("RGB")
    inp = _processor(rgb).unsqueeze(0).to(_device)

    with torch.no_grad():
        result = _model(inp)

    mask = result[0][0].squeeze().cpu().numpy()
    mask = (mask * 255).clip(0, 255).astype("uint8")
    mask_img = Image.fromarray(mask).resize((W, H), Image.BILINEAR)

    out = np.array(rgb.convert("RGBA"))
    out[:, :, 3] = np.array(mask_img)
    return Image.fromarray(out, "RGBA")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "bg_remover_server",
        "endpoints": ["/health", "/remove-bg/image"],
        "device": _device,
    }


@app.get("/health")
def health():
    return JSONResponse(
        {
            "status": "ok" if _model is not None else "loading",
            "device": _device,
            "model": MODEL_ID,
        }
    )


@app.post("/remove-bg/image")
async def remove_bg_image(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max {MAX_UPLOAD_MB} MB.",
        )

    try:
        pil_in = Image.open(io.BytesIO(data))
        pil_in.load()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Not a valid image: {e}")

    try:
        pil_out = _remove_bg(pil_in)
    except Exception as e:
        log.exception("Background removal failed")
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

    buf = io.BytesIO()
    pil_out.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "bg_remover_server:app",
        host=HOST,
        port=PORT,
        log_level="info",
    )
