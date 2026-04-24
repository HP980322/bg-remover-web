# BG Remover — Blender Add-on

Remove backgrounds from images directly inside Blender using the [RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4) AI model.

Single file. No system Python. No virtual environment. No torch, no transformers. Auto-installs everything it needs.

Produces output **byte-identical** to [the web version](https://hp980322.github.io/bg-remover-web/) — same model, same preprocessing, same edge cleanup.

## Installation

1. Download [`bg_remover_addon.py`](bg_remover_addon.py).
2. In Blender: **Edit → Preferences → Add-ons → Install...** and select the file.
3. Tick **Image: BG Remover** to enable it.
4. The add-on will start auto-installing dependencies in the background. You'll see a "Working… Installing…" box appear in the BG Remover panel — Blender stays responsive throughout.

That's it. No further setup.

You'll find the **BG Remover** tab in the **Image Editor** N-panel (press **N** in the Image Editor to show it).

### What gets installed

On first enable, the add-on installs into Blender's bundled Python (`%APPDATA%\Python\Python311\site-packages` on Windows):

- `onnxruntime==1.20.1` — runs the AI model
- `numpy` — array math
- `scipy` — connected components, morphology
- `Pillow` — image I/O

Total disk: ~150 MB of Python packages plus a 176 MB ONNX model file (cached at `<addons-dir>/bg_remover_data/rmbg_1_4.onnx`).

The 176 MB model downloads once from Hugging Face and is reused forever after.

## Usage

| Button | What it does |
|--------|--------------|
| **Remove Background** | Processes the image currently open in the Image Editor |
| **Remove BG from Render** | Processes the latest Render Result |
| **Process Image Sequence** | Batch-processes a folder of images → saves as PNG sequence with `_nobg` suffix |

Output: a new image with the same RGB and the AI-predicted alpha as the alpha channel. The original is untouched.

### Workflow examples

**Removing background from a render:**
1. Render normally (F12).
2. Click **Remove BG from Render** in the BG Remover panel.
3. The result appears as `Render_nobg` — usable in the Compositor via an Image node.

**Batch-processing an animation:**
1. Render to a PNG sequence (`/path/to/anim/frame_0001.png`, `frame_0002.png`, …).
2. Click **Process Image Sequence** → pick the folder.
3. Output goes to `/path/to/anim_nobg/frame_0001_nobg.png`, etc.
4. Import as an Image Sequence in the VSE or Compositor.

**Texture work:**
1. Open a texture in the Image Editor.
2. **Remove Background** → result appears as `<name>_nobg`.
3. Use it in your material with alpha blend mode.

## Performance

Inference runs on **CPU** by default. For typical images:

| Image size | Time per image |
|------------|----------------|
| 512×512 | ~0.3 s |
| 1024×1024 | ~1.2 s |
| 1920×1080 | ~3.8 s |

(Modern CPUs. The bulk of the time is the cleanMask post-processing; raw AI inference is ~30 ms.)

GPU (NVIDIA CUDA) acceleration is not enabled by default because:
- The CUDA build of onnxruntime requires CUDA 12 + cuDNN 9 installed system-wide.
- It only helps NVIDIA users.
- For one-off image edits, CPU is plenty fast.

If you want GPU, install `onnxruntime-gpu` manually into Blender's Python and the add-on will pick it up:

```
"<blender>\python\bin\python.exe" -m pip install --user --force-reinstall onnxruntime-gpu==1.20.1
```

## Settings

Most users never touch these. They live in **Edit → Preferences → Add-ons → BG Remover**.

| Setting | Default | What it does |
|---------|---------|--------------|
| **Auto-install dependencies on startup** | On | If something's missing when Blender starts, install it automatically. Turn off for manual control. |
| **Show advanced tools in panel** | Off | Add a collapsible "Advanced" section to the panel with Recheck, Debug Info, Force Reinstall, and Re-download Model. |
| **Refine edges and remove noise** | On | The cleanMask post-processor. Adds ~1 second per image but produces visibly cleaner edges and removes stray pixels. Turn off for ~30 ms processing time when batching hundreds of frames. |

## Troubleshooting

### `Cannot import: onnxruntime` with "DLL load failed" or "动态链接库" error (Windows only)

This is an onnxruntime DLL initialization failure, almost always caused by a missing Microsoft Visual C++ Redistributable.

**Fix:**
1. The add-on detects this case and shows a button: **Install vc_redist.x64.exe** — click it.
2. Run the installer (~25 MB, takes 30 seconds).
3. **Close Blender completely.**
4. Reopen Blender. The add-on should import onnxruntime cleanly now.

If the error persists after installing VC++ redist, your antivirus may be quarantining onnxruntime's DLL files. Check Windows Defender / 360 / etc. quarantine and allow the files in `<blender>\python\Lib\site-packages\onnxruntime`.

Or click **Force Reinstall onnxruntime** in the addon's Advanced section — that wipes and reinstalls the package fresh.

### Install Dependencies button is greyed out / nothing happens

Check the panel: if it says "Working…" with a clock icon, an install is already running in the background. Wait for it to finish (typically 30-60 seconds).

### Output looks rougher than the web version

Check **Edit → Preferences → Add-ons → BG Remover → Output quality → Refine edges and remove noise** is on. With it on, the addon's output is byte-identical to the web version.

### Want the original error message for a problem

In the panel, enable **Show advanced tools in panel** in addon prefs, then click **Debug Info** in the Advanced section. It writes a full report to `<addons-dir>/bg_remover_data/diagnostics.txt`. Click **Open Debug Report** to view it.

## How it works

For the technical curious:

1. The image is resized to 1024×1024 with bilinear interpolation.
2. Pixels are normalized: `(x / 255 - 0.5) / 1.0` per channel, transposed to NCHW float32.
3. RMBG-1.4 (ONNX) produces a 1024×1024 single-channel foreground probability mask.
4. Mask is min-max normalized to [0, 1] and resized back to source dimensions with bilinear interpolation.
5. Optional `cleanMask v16` post-processor (Python port of the JS algorithm in `../index.html`):
   - K-means (K=8, 20 iterations) background color model from edge-pixel samples.
   - Per-pixel nearest-cluster distance + per-cluster tolerance for `is_bg(p)`.
   - Binarize at threshold 127, drop components < 0.5% of image area.
   - Iterative refinement (up to 8 passes): handle sky → fill background holes → morph close ×2 → handle sky → grow into body-colored pixels → handle sky.
   - Per-row thin-sky-stripe removal (runs <20 px, bluish, mostly background).
   - Drop components <0.3% of area.
   - 3×3 box blur with hard 0/255 interior, anti-aliased edges.
6. Output: original RGB + AI-predicted alpha, packed as RGBA PNG into Blender's image data.

The cleanMask Python port is verified pixel-identical against the JavaScript reference on 150,000+ test pixels.

## License

Add-on code: do whatever you want, attribution appreciated.

The RMBG-1.4 model is by Bria AI under their own license — see [their model card](https://huggingface.co/briaai/RMBG-1.4) before commercial use.

## Issues / feedback

[Open an issue](https://github.com/HP980322/bg-remover-web/issues) on the repo.
