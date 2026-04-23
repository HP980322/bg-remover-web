# ============================================================
# BG Remover — Blender Add-on
# Removes backgrounds from images using RMBG-1.4 AI.
#
# Runs entirely inside Blender's bundled Python via ONNX Runtime.
# No system Python, no venv, no torch, no transformers. Just works.
#
# On first use the addon installs onnxruntime + Pillow + numpy into
# Blender's Python (~80 MB) and downloads the RMBG-1.4 ONNX model
# (~176 MB, cached forever).
#
# Installation:
#   Edit > Preferences > Add-ons > Install > select this file
#   Enable "Image: BG Remover"
#
# Usage:
#   Image Editor > N-panel > BG Remover tab
# ============================================================

bl_info = {
    "name": "BG Remover",
    "author": "HP980322",
    "version": (3, 0, 0),
    "blender": (3, 0, 0),
    "location": "Image Editor > Sidebar > BG Remover",
    "description": "Remove image/render background using RMBG-1.4 AI (ONNX Runtime)",
    "category": "Image",
}

import bpy
import os
import sys
import site
import shutil
import tempfile
import importlib
import subprocess
import urllib.request
import urllib.error
from pathlib import Path


# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_URL = "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model.onnx"
MODEL_FILENAME = "rmbg_1_4.onnx"
MODEL_INPUT_SIZE = 1024

REQUIRED_PACKAGES = ["onnxruntime", "numpy", "Pillow"]


# ── Globals ────────────────────────────────────────────────────────────────────

_session = None
_session_provider = None
_last_error = None


def _addon_data_dir():
    base = Path(bpy.utils.user_resource('SCRIPTS', path='addons'))
    d = base / 'bg_remover_data'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _model_path():
    return _addon_data_dir() / MODEL_FILENAME


def _refresh_sys_path():
    try:
        for p in site.getsitepackages() + [site.getusersitepackages()]:
            if p and p not in sys.path:
                sys.path.append(p)
        importlib.invalidate_caches()
    except Exception as e:
        print(f"[BG Remover] sys.path refresh warning: {e}")


def _check_packages():
    missing = []
    checks = [("onnxruntime", "onnxruntime"),
              ("numpy", "numpy"),
              ("Pillow", "PIL")]
    for pip_name, import_name in checks:
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pip_name)
    return missing


def ensure_deps():
    missing = _check_packages()
    if not missing:
        return True

    python = sys.executable
    print(f"[BG Remover] Installing: {', '.join(missing)}")

    try:
        subprocess.check_call(
            [python, '-m', 'ensurepip', '--upgrade'],
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        )
    except Exception:
        pass

    try:
        result = subprocess.run(
            [python, '-m', 'pip', 'install', '--upgrade', '--user'] + missing,
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"pip install failed:\n{(result.stderr or result.stdout)[-1500:]}\n\n"
                f"Try running Blender as Administrator (Windows) / with sudo "
                f"(Linux/macOS), or install manually with:\n"
                f"  {python} -m pip install --user {' '.join(missing)}"
            )
    except FileNotFoundError:
        raise RuntimeError(
            "Blender's Python has no pip and ensurepip failed. "
            "Please reinstall Blender."
        )

    _refresh_sys_path()

    still_missing = _check_packages()
    if still_missing:
        raise RuntimeError(
            f"Installed packages but can't import: {', '.join(still_missing)}. "
            "Try restarting Blender."
        )
    return True


def _download_with_progress(url, dst_path):
    tmp_path = dst_path.with_suffix('.part')
    print(f"[BG Remover] Downloading RMBG-1.4 ONNX model (~176 MB) from {url}")
    last_report = 0
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            total = int(resp.headers.get('Content-Length', 0))
            downloaded = 0
            with open(tmp_path, 'wb') as f:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total and downloaded - last_report > 5 * 1024 * 1024:
                        pct = 100 * downloaded // total
                        mb = downloaded / (1024 * 1024)
                        print(f"[BG Remover] {pct}% ({mb:.1f} MB)")
                        last_report = downloaded
        tmp_path.rename(dst_path)
        print(f"[BG Remover] Model saved to {dst_path}")
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise


def ensure_model():
    mp = _model_path()
    if mp.is_file() and mp.stat().st_size > 100 * 1024 * 1024:
        return mp
    _download_with_progress(MODEL_URL, mp)
    return mp


def get_session():
    global _session, _session_provider
    if _session is not None:
        return _session

    ensure_deps()
    model_path = ensure_model()

    import onnxruntime as ort

    available = ort.get_available_providers()
    if 'CUDAExecutionProvider' in available:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    _session = ort.InferenceSession(str(model_path), providers=providers)
    _session_provider = _session.get_providers()[0]
    print(f"[BG Remover] ONNX session ready ({_session_provider})")
    return _session


def remove_bg(pil_img):
    """Run RMBG-1.4 on a PIL Image. Returns RGBA PIL Image.

    Preprocessing matches briaai's reference code exactly:
    resize 1024x1024 bilinear -> /255 -> (x-0.5)/1.0 -> NCHW float32."""
    import numpy as np
    from PIL import Image

    session = get_session()
    input_name = session.get_inputs()[0].name

    src_rgb = pil_img.convert('RGB')
    W, H = src_rgb.size

    resized = src_rgb.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 1.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0).astype(np.float32)

    outputs = session.run(None, {input_name: arr})
    mask = np.asarray(outputs[0]).squeeze()
    if mask.ndim != 2:
        raise RuntimeError(f"Unexpected mask shape from model: {mask.shape}")

    mn, mx = float(mask.min()), float(mask.max())
    if mx > mn:
        mask = (mask - mn) / (mx - mn)
    else:
        mask = np.zeros_like(mask)

    mask_img = Image.fromarray((mask * 255).clip(0, 255).astype('uint8'))
    mask_img = mask_img.resize((W, H), Image.BILINEAR)

    rgba = np.asarray(src_rgb).copy()
    rgba = np.dstack([rgba, np.asarray(mask_img)])
    return Image.fromarray(rgba, 'RGBA')


# ── Blender image helpers ─────────────────────────────────────────────────────

def blender_image_to_pil(image):
    ensure_deps()
    from PIL import Image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tf:
        tmp = tf.name
    try:
        image.save_render(tmp)
        return Image.open(tmp).copy()
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def pil_to_blender_image(pil_img, name):
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tf:
        tmp = tf.name
    try:
        pil_img.save(tmp, 'PNG')
        img = bpy.data.images.load(tmp)
        img.name = name
        img.pack()
        return img
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ── Operators ─────────────────────────────────────────────────────────────────

class BGRemover_OT_InstallDeps(bpy.types.Operator):
    """Install onnxruntime + numpy + Pillow into Blender's Python,
    then download the RMBG-1.4 ONNX model (~176 MB)."""
    bl_idname = 'bgremover.install_deps'
    bl_label = 'Install Dependencies'

    def execute(self, context):
        global _last_error
        _last_error = None
        try:
            self.report({'INFO'}, 'Installing packages…')
            ensure_deps()
            self.report({'INFO'}, 'Downloading model if needed…')
            ensure_model()
            get_session()
            self.report({'INFO'}, '✅ Ready!')
        except Exception as e:
            _last_error = str(e)
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        return {'FINISHED'}


class BGRemover_OT_RemoveBackground(bpy.types.Operator):
    """Remove background from the active image in the Image Editor"""
    bl_idname = 'bgremover.remove_background'
    bl_label = 'Remove Background'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (context.area and
                context.area.type == 'IMAGE_EDITOR' and
                context.area.spaces.active.image is not None)

    def execute(self, context):
        global _last_error
        _last_error = None

        image = context.area.spaces.active.image
        out_name = Path(image.name).stem + '_nobg'
        area = context.area

        self.report({'INFO'}, f"Processing '{image.name}'…")
        try:
            pil_in = blender_image_to_pil(image)
            pil_out = remove_bg(pil_in)
            result_img = pil_to_blender_image(pil_out, out_name)
            area.spaces.active.image = result_img
            self.report({'INFO'}, f"Done! Saved as '{out_name}'")
        except Exception as e:
            _last_error = str(e)
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        return {'FINISHED'}


class BGRemover_OT_RemoveFromRender(bpy.types.Operator):
    """Remove background from the most recent Render Result"""
    bl_idname = 'bgremover.remove_from_render'
    bl_label = 'Remove BG from Render'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bpy.data.images.get('Render Result') is not None

    def execute(self, context):
        global _last_error
        _last_error = None

        render_img = bpy.data.images['Render Result']
        out_name = 'Render_nobg'

        self.report({'INFO'}, 'Processing render…')
        try:
            pil_in = blender_image_to_pil(render_img)
            pil_out = remove_bg(pil_in)
            result_img = pil_to_blender_image(pil_out, out_name)
            for area in context.screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    area.spaces.active.image = result_img
                    break
            self.report({'INFO'}, f"Done! Saved as '{out_name}'")
        except Exception as e:
            _last_error = str(e)
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        return {'FINISHED'}


class BGRemover_OT_ProcessSequence(bpy.types.Operator):
    """Remove background from every frame of an image sequence"""
    bl_idname = 'bgremover.process_sequence'
    bl_label = 'Process Image Sequence'
    bl_options = {'REGISTER'}

    directory: bpy.props.StringProperty(name='Input Folder', subtype='DIR_PATH')
    output_dir: bpy.props.StringProperty(name='Output Folder', subtype='DIR_PATH')

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        global _last_error
        _last_error = None

        src = Path(self.directory)
        out = Path(self.output_dir) if self.output_dir else src.parent / (src.name + '_nobg')
        out.mkdir(parents=True, exist_ok=True)

        exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff')
        frames = sorted([f for f in src.iterdir() if f.suffix.lower() in exts])
        if not frames:
            self.report({'ERROR'}, 'No image files found.')
            return {'CANCELLED'}

        self.report({'INFO'}, f'Processing {len(frames)} frames — watch console.')

        try:
            ensure_deps()
            from PIL import Image
            get_session()
        except Exception as e:
            _last_error = str(e)
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        ok = 0
        for i, frame in enumerate(frames):
            try:
                pil_in = Image.open(frame)
                pil_out = remove_bg(pil_in)
                pil_out.save(out / (frame.stem + '_nobg.png'), 'PNG')
                print(f'[BG Remover] {i + 1}/{len(frames)} {frame.name}')
                ok += 1
            except Exception as e:
                print(f'[BG Remover] Skipped {frame.name}: {e}')

        print(f'[BG Remover] ✅ Done! {ok}/{len(frames)} frames -> {out}')
        self.report({'INFO'}, f'Done! {ok}/{len(frames)} frames -> {out}')
        return {'FINISHED'}


class BGRemover_OT_ClearModel(bpy.types.Operator):
    """Delete the cached ONNX model so it re-downloads next time"""
    bl_idname = 'bgremover.clear_model'
    bl_label = 'Re-download Model'

    def execute(self, context):
        global _session, _session_provider
        mp = _model_path()
        if mp.exists():
            try:
                mp.unlink()
            except Exception as e:
                self.report({'ERROR'}, f'Could not delete {mp}: {e}')
                return {'CANCELLED'}
        _session = None
        _session_provider = None
        self.report({'INFO'}, 'Model cache cleared.')
        return {'FINISHED'}


# ── Panel ──────────────────────────────────────────────────────────────────────

class BGRemover_PT_Panel(bpy.types.Panel):
    bl_label = 'BG Remover'
    bl_idname = 'BGREMOVER_PT_panel'
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'BG Remover'

    def draw(self, context):
        layout = self.layout

        missing = _check_packages()
        model_exists = _model_path().is_file()

        if missing:
            box = layout.box()
            box.alert = True
            box.label(text='Setup required', icon='ERROR')
            box.label(text=f"Missing: {', '.join(missing)}")
            box.operator('bgremover.install_deps', icon='IMPORT')
            return

        if not model_exists:
            box = layout.box()
            box.label(text='Model not downloaded', icon='IMPORT')
            box.label(text='~176 MB, one-time', icon='INFO')
            box.operator('bgremover.install_deps',
                         text='Download Model', icon='URL')
            return

        if _session is not None:
            provider_short = (_session_provider or '').replace('ExecutionProvider', '')
            layout.label(text=f'✅ Ready ({provider_short})', icon='CHECKMARK')
        else:
            layout.label(text='Ready — model loads on first use', icon='TIME')

        if _last_error:
            err_box = layout.box()
            err_box.alert = True
            first_line = _last_error.strip().splitlines()[0][:60]
            err_box.label(text=f"Last error: {first_line}", icon='ERROR')

        layout.separator()

        space = context.area.spaces.active if context.area else None
        if space and space.image:
            box = layout.box()
            box.label(text=space.image.name, icon='IMAGE_DATA')
            box.operator('bgremover.remove_background', icon='MATFLUID')
        else:
            layout.label(text='Open an image first', icon='INFO')

        layout.separator()

        if bpy.data.images.get('Render Result'):
            layout.operator('bgremover.remove_from_render', icon='RENDER_STILL')
        else:
            row = layout.row()
            row.enabled = False
            row.operator('bgremover.remove_from_render', icon='RENDER_STILL')

        layout.operator('bgremover.process_sequence', icon='SEQUENCE')

        layout.separator()
        layout.operator('bgremover.clear_model', icon='TRASH')


# ── Registration ───────────────────────────────────────────────────────────────

classes = [
    BGRemover_OT_InstallDeps,
    BGRemover_OT_RemoveBackground,
    BGRemover_OT_RemoveFromRender,
    BGRemover_OT_ProcessSequence,
    BGRemover_OT_ClearModel,
    BGRemover_PT_Panel,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    global _session, _session_provider
    _session = None
    _session_provider = None
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == '__main__':
    register()
