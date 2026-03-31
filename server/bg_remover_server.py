#!/usr/bin/env python3
"""
bg_remover_server.py  —  FastAPI background removal server
Handles: single image, batch images, video frame-by-frame
Used by: web clients, Blender add-on

Install:
    pip install fastapi uvicorn python-multipart pillow transformers torch torchvision opencv-python-headless

Run:
    python bg_remover_server.py
    # or: uvicorn bg_remover_server:app --host 0.0.0.0 --port 8000
"""

import io, os, time, zipfile, tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import Response, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Model loading ─────────────────────────────────────────────────────
print("Loading RMBG-1.4 model…")
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = AutoModelForImageSegmentation.from_pretrained(
    "briaai/RMBG-1.4", trust_remote_code=True
)
model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
])

print("Model ready.")

# ── FastAPI app ───────────────────────────────────────────────────────
app = FastAPI(title="BG Remover API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def remove_bg_pil(img: Image.Image) -> Image.Image:
    """Run RMBG-1.4 on a PIL image, return RGBA PIL image."""
    W, H = img.size
    rgb = img.convert("RGB")
    inp = transform(rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        result = model(inp)

    # result[0][0] is the mask tensor
    mask = result[0][0].squeeze().cpu().numpy()
    mask = (mask * 255).clip(0, 255).astype(np.uint8)

    mask_img = Image.fromarray(mask).resize((W, H), Image.BILINEAR)
    mask_arr = np.array(mask_img)

    out = np.array(rgb.convert("RGBA"))
    out[:, :, 3] = mask_arr
    return Image.fromarray(out, "RGBA")


def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "model": "briaai/RMBG-1.4", "device": device}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/remove-bg/image")
async def remove_bg_image(file: UploadFile = File(...)):
    """
    Remove background from a single image.
    Returns PNG with transparent background.
    """
    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Could not read image: {e}")

    result = remove_bg_pil(img)
    png_bytes = pil_to_png_bytes(result)

    stem = Path(file.filename or "image").stem
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{stem}_nobg.png"'},
    )


@app.post("/remove-bg/batch")
async def remove_bg_batch(files: list[UploadFile] = File(...)):
    """
    Remove background from multiple images.
    Returns a ZIP containing all processed PNGs.
    """
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            try:
                data = await f.read()
                img = Image.open(io.BytesIO(data)).convert("RGB")
                result = remove_bg_pil(img)
                png_bytes = pil_to_png_bytes(result)
                stem = Path(f.filename or "image").stem
                zf.writestr(f"{stem}_nobg.png", png_bytes)
            except Exception as e:
                zf.writestr(f"{f.filename}.error.txt", str(e))

    return Response(
        content=zip_buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="batch_nobg.zip"'},
    )


# Video job store (simple in-memory)
video_jobs: dict = {}

@app.post("/remove-bg/video")
async def remove_bg_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    fps: Optional[int] = None,
):
    """
    Remove background from every frame of a video.
    Starts async job, returns job_id.
    Poll GET /remove-bg/video/{job_id} for status/result.
    """
    try:
        import cv2
    except ImportError:
        raise HTTPException(500, "opencv-python-headless not installed. Run: pip install opencv-python-headless")

    data = await file.read()
    job_id = f"job_{int(time.time()*1000)}"
    video_jobs[job_id] = {"status": "queued", "progress": 0, "total": 0}

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_in.write(data)
    tmp_in.close()

    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp_out.close()

    background_tasks.add_task(
        _process_video_job, job_id, tmp_in.name, tmp_out.name, fps
    )
    return {"job_id": job_id}


def _process_video_job(job_id: str, in_path: str, out_path: str, fps_limit):
    import cv2
    job = video_jobs[job_id]
    try:
        job["status"] = "processing"
        cap = cv2.VideoCapture(in_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 24
        job["total"] = total

        skip = max(1, int(round(src_fps / fps_limit))) if fps_limit else 1

        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % skip == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb)
                    result = remove_bg_pil(pil)
                    png = pil_to_png_bytes(result)
                    zf.writestr(f"frame_{idx:06d}.png", png)
                job["progress"] = idx + 1
                idx += 1
        cap.release()

        job["status"] = "done"
        job["result_path"] = out_path
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
    finally:
        os.unlink(in_path)


@app.get("/remove-bg/video/{job_id}")
def video_job_status(job_id: str):
    job = video_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] == "done":
        return FileResponse(
            job["result_path"],
            media_type="application/zip",
            filename=f"{job_id}_frames.zip",
        )
    return JSONResponse(job)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
