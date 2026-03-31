# BG Remover — Blender Add-on

Removes backgrounds from images directly inside Blender using the RMBG-1.4 AI model.

## Installation

1. Download `bg_remover_addon.py`
2. In Blender: **Edit → Preferences → Add-ons → Install**
3. Select the `.py` file, then tick **Image: BG Remover** to enable it
4. Find the **BG Remover** tab in the **Image Editor** N-panel (press **N**)

## Two modes

### Local AI mode (default)
Runs RMBG-1.4 directly inside Blender's Python. No internet needed after first download.

- First use: click **Install AI Dependencies** in Preferences — this installs `torch`, `transformers`, and `torchvision` into Blender's Python (~1GB, one-time)
- Model downloads from HuggingFace on first use (~175 MB, cached forever)
- GPU auto-detected if available (CUDA), otherwise runs on CPU

### Server mode
Sends images to a running `bg_remover_server.py` — faster if you have a GPU server.

- Set the server URL in Preferences (default: `http://localhost:8000`)
- Start the server: `python server/bg_remover_server.py`

## Features

| Button | What it does |
|--------|--------------|
| **Remove Background** | Processes the active image in the Image Editor |
| **Remove BG from Render** | Processes the latest Render Result |
| **Process Image Sequence** | Batch-processes a folder of images (PNG/JPG/EXR) → saves as PNG sequence |

## Workflow tips

**For video / animation:**
1. Render your animation to a folder (PNG sequence)
2. Use **Process Image Sequence** → select the render output folder
3. The add-on saves `frame_000001_nobg.png`, `frame_000002_nobg.png`, etc.
4. In the VSE or compositor, import as an Image Sequence

**For materials:**
1. Open a texture in the Image Editor
2. Click **Remove Background** → the result image appears
3. Use it as a texture with alpha blend in your material

**For compositing:**
1. Render → click **Remove BG from Render**
2. The result is available as `Render_nobg` in your image data
3. Use it in the Compositor with an Image node
