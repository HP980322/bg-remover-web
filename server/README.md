# BG Remover — FastAPI Server + Blender Add-on

Removes backgrounds from images and videos using RMBG-1.4, locally on your machine.

## Quick start

```bash
# 1. Install dependencies
pip install fastapi uvicorn python-multipart pillow transformers torch torchvision opencv-python-headless

# 2. Start the server
python bg_remover_server.py
# Server runs at http://localhost:8000
```

---

## API endpoints

| Method | Endpoint | Input | Output |
|--------|----------|-------|--------|
| POST | `/remove-bg/image` | image file | PNG with alpha |
| POST | `/remove-bg/batch` | multiple images | ZIP of PNGs |
| POST | `/remove-bg/video` | video file | `{job_id}` |
| GET | `/remove-bg/video/{job_id}` | — | ZIP of frame PNGs (or status JSON) |
| GET | `/health` | — | `{"status":"ok"}` |

### Single image (curl)
```bash
curl -X POST http://localhost:8000/remove-bg/image \
  -F "file=@photo.jpg" \
  --output photo_nobg.png
```

### Video (curl)
```bash
# Start async job (fps=10 means process every 2-3 frames of a 24fps video)
curl -X POST http://localhost:8000/remove-bg/video \
  -F "file=@clip.mp4" -F "fps=10"
# Returns: {"job_id": "job_1234567"}

# Poll until status=done, then get ZIP of PNG frames
curl http://localhost:8000/remove-bg/video/job_1234567 --output frames.zip
```

### Python example
```python
import requests

with open("photo.jpg", "rb") as f:
    r = requests.post("http://localhost:8000/remove-bg/image", files={"file": f})
with open("photo_nobg.png", "wb") as f:
    f.write(r.content)
```

---

## Blender Add-on

### Install
1. Start `bg_remover_server.py` first
2. In Blender: **Edit → Preferences → Add-ons → Install**
3. Select `bg_remover_addon.py`, tick the checkbox to enable
4. Go to the **Image Editor**, press **N** to open the sidebar
5. Find the **BG Remover** tab

### Usage
- **Remove Background** — sends the currently open image to the server, loads the result back
- **Remove BG from Render** — works on the latest render result
- **Test Connection** — checks the server is reachable

### Use cases
- Clean reference images
- Remove BG from texture photos before applying to materials  
- Process rendered images for compositing
- Batch process images via the API directly
