# ============================================================
# BG Remover — Blender Add-on
# Removes backgrounds from images using RMBG-1.4 AI
#
# Two modes:
#   1. LOCAL  — runs RMBG-1.4 directly inside Blender's Python
#               (auto-installs transformers + torch on first use)
#   2. SERVER — sends images to your running bg_remover_server.py
#               (faster if you have a GPU server)
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
    "version": (2, 0, 0),
    "blender": (3, 0, 0),
    "location": "Image Editor > Sidebar > BG Remover",
    "description": "Remove image/render background using RMBG-1.4 AI (local or server)",
    "category": "Image",
}

import bpy
import os
import io
import sys
import math
import struct
import tempfile
import subprocess
import urllib.request
import urllib.error
from pathlib import Path


# ── Preferences ───────────────────────────────────────────────────────────────

class BGRemoverPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    mode: bpy.props.EnumProperty(
        name="Mode",
        items=[
            ('LOCAL',  'Local (built-in AI)',  'Run RMBG-1.4 directly in Blender — no server needed'),
            ('SERVER', 'Server',               'Send images to a running bg_remover_server.py'),
        ],
        default='LOCAL',
    )
    server_url: bpy.props.StringProperty(
        name="Server URL",
        default="http://localhost:8000",
        description="URL of the running bg_remover_server.py (Server mode only)",
    )
    model_loaded: bpy.props.BoolProperty(default=False)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "mode", expand=True)
        if self.mode == 'SERVER':
            layout.prop(self, "server_url")
            layout.operator("bgremover.test_connection", icon="LINKED")
        else:
            row = layout.row()
            row.operator("bgremover.install_deps", icon="IMPORT")


# ── Globals ────────────────────────────────────────────────────────────────────

_model = None
_processor = None
_device = 'cpu'


def get_prefs(context):
    return context.preferences.addons[__name__].preferences


def ensure_deps():
    """Install transformers + torch into Blender's Python if missing."""
    python = sys.executable
    missing = []
    try:
        import torch
    except ImportError:
        missing.append('torch')
    try:
        import transformers
    except ImportError:
        missing.append('transformers')
    try:
        import torchvision
    except ImportError:
        missing.append('torchvision')
    try:
        import PIL
    except ImportError:
        missing.append('Pillow')

    if missing:
        subprocess.check_call(
            [python, '-m', 'pip', 'install', '--upgrade'] + missing,
            stdout=subprocess.DEVNULL
        )
    return True


def load_model():
    global _model, _processor, _device
    if _model is not None:
        return True
    try:
        ensure_deps()
        import torch
        from transformers import AutoModelForImageSegmentation
        from torchvision import transforms

        _device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _model = AutoModelForImageSegmentation.from_pretrained(
            'briaai/RMBG-1.4', trust_remote_code=True
        )
        _model.to(_device).eval()
        _processor = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
        ])
        return True
    except Exception as e:
        print(f"[BG Remover] Model load failed: {e}")
        return False


# ── Core: remove bg from PIL Image (local) ────────────────────────────────────

def remove_bg_local(pil_img):
    """Run RMBG-1.4 on a PIL RGB image, return RGBA PIL image."""
    if not load_model():
        raise RuntimeError("Could not load RMBG-1.4 model. Check console for details.")

    import torch
    import numpy as np
    from PIL import Image

    W, H = pil_img.size
    rgb = pil_img.convert('RGB')
    inp = _processor(rgb).unsqueeze(0).to(_device)

    with torch.no_grad():
        result = _model(inp)

    mask = result[0][0].squeeze().cpu().numpy()
    mask = (mask * 255).clip(0, 255).astype('uint8')
    mask_img = Image.fromarray(mask).resize((W, H), Image.BILINEAR)

    out = np.array(rgb.convert('RGBA'))
    out[:, :, 3] = np.array(mask_img)
    return Image.fromarray(out, 'RGBA')


# ── Core: remove bg via server ────────────────────────────────────────────────

def remove_bg_server(server_url, png_bytes, filename):
    """POST image to /remove-bg/image, return PNG bytes."""
    boundary = '----BGRemoverBlender'
    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f'Content-Type: image/png\r\n\r\n'
    ).encode() + png_bytes + f'\r\n--{boundary}--\r\n'.encode()

    req = urllib.request.Request(
        f"{server_url}/remove-bg/image",
        data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


# ── Blender image helpers ─────────────────────────────────────────────────────

def blender_image_to_pil(image):
    """Convert a Blender image to a PIL RGB image via temp PNG."""
    from PIL import Image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tf:
        tmp = tf.name
    try:
        image.save_render(tmp)
        return Image.open(tmp).copy()
    finally:
        os.unlink(tmp)


def blender_image_to_png_bytes(image):
    """Save Blender image to PNG bytes."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tf:
        tmp = tf.name
    try:
        image.save_render(tmp)
        with open(tmp, 'rb') as f:
            return f.read()
    finally:
        os.unlink(tmp)


def pil_to_blender_image(pil_img, name):
    """Load a PIL RGBA image as a new Blender image, packed into .blend."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tf:
        pil_img.save(tf, 'PNG')
        tmp = tf.name
    try:
        img = bpy.data.images.load(tmp)
        img.name = name
        img.pack()
        return img
    finally:
        os.unlink(tmp)


def png_bytes_to_blender_image(png_bytes, name):
    """Load raw PNG bytes as a new Blender image, packed."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tf:
        tf.write(png_bytes)
        tmp = tf.name
    try:
        img = bpy.data.images.load(tmp)
        img.name = name
        img.pack()
        return img
    finally:
        os.unlink(tmp)


# ── Operators ─────────────────────────────────────────────────────────────────

class BGRemover_OT_InstallDeps(bpy.types.Operator):
    """Install required Python packages (torch, transformers) into Blender"""
    bl_idname = 'bgremover.install_deps'
    bl_label = 'Install AI Dependencies'

    def execute(self, context):
        try:
            self.report({'INFO'}, 'Installing packages — this may take a few minutes…')
            ensure_deps()
            self.report({'INFO'}, '✅ Dependencies installed! Model will load on first use.')
        except Exception as e:
            self.report({'ERROR'}, f'Install failed: {e}')
        return {'FINISHED'}


class BGRemover_OT_TestConnection(bpy.types.Operator):
    """Test connection to the background removal server"""
    bl_idname = 'bgremover.test_connection'
    bl_label = 'Test Connection'

    def execute(self, context):
        prefs = get_prefs(context)
        try:
            req = urllib.request.Request(f"{prefs.server_url}/health", method='GET')
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    self.report({'INFO'}, f'✅ Connected to {prefs.server_url}')
                    return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f'❌ Cannot connect to {prefs.server_url}: {e}')
        return {'CANCELLED'}


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
        prefs = get_prefs(context)
        image = context.area.spaces.active.image
        out_name = Path(image.name).stem + '_nobg'

        self.report({'INFO'}, f"Processing '{image.name}'…")

        try:
            if prefs.mode == 'LOCAL':
                pil_in = blender_image_to_pil(image)
                pil_out = remove_bg_local(pil_in)
                result_img = pil_to_blender_image(pil_out, out_name)
            else:
                png_bytes = blender_image_to_png_bytes(image)
                result_bytes = remove_bg_server(prefs.server_url, png_bytes, image.name + '.png')
                result_img = png_bytes_to_blender_image(result_bytes, out_name)
        except urllib.error.URLError as e:
            self.report({'ERROR'}, f"Cannot reach server: {e.reason}")
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        context.area.spaces.active.image = result_img
        self.report({'INFO'}, f"Done! Saved as '{out_name}'")
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
        prefs = get_prefs(context)
        render_img = bpy.data.images['Render Result']
        out_name = 'Render_nobg'

        self.report({'INFO'}, 'Processing render…')
        try:
            if prefs.mode == 'LOCAL':
                pil_in = blender_image_to_pil(render_img)
                pil_out = remove_bg_local(pil_in)
                result_img = pil_to_blender_image(pil_out, out_name)
            else:
                png_bytes = blender_image_to_png_bytes(render_img)
                result_bytes = remove_bg_server(prefs.server_url, png_bytes, 'render.png')
                result_img = png_bytes_to_blender_image(result_bytes, out_name)
        except urllib.error.URLError as e:
            self.report({'ERROR'}, f"Cannot reach server: {e.reason}")
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        # Show result in whichever Image Editor is open
        for area in context.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                area.spaces.active.image = result_img
                break

        self.report({'INFO'}, f"Done! Saved as '{out_name}'")
        return {'FINISHED'}


class BGRemover_OT_ProcessSequence(bpy.types.Operator):
    """Remove background from every frame of an image sequence, save as PNG sequence"""
    bl_idname = 'bgremover.process_sequence'
    bl_label = 'Process Image Sequence'
    bl_options = {'REGISTER'}

    directory: bpy.props.StringProperty(
        name='Input Folder',
        subtype='DIR_PATH',
    )
    output_dir: bpy.props.StringProperty(
        name='Output Folder',
        subtype='DIR_PATH',
    )

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        prefs = get_prefs(context)
        src = Path(self.directory)
        out = Path(self.output_dir) if self.output_dir else src.parent / (src.name + '_nobg')
        out.mkdir(parents=True, exist_ok=True)

        frames = sorted([f for f in src.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.exr', '.tif')])
        if not frames:
            self.report({'ERROR'}, 'No image files found in selected folder.')
            return {'CANCELLED'}

        self.report({'INFO'}, f'Processing {len(frames)} frames…')
        from PIL import Image

        for i, frame in enumerate(frames):
            try:
                if prefs.mode == 'LOCAL':
                    pil_in = Image.open(frame).convert('RGB')
                    pil_out = remove_bg_local(pil_in)
                    pil_out.save(out / (frame.stem + '_nobg.png'), 'PNG')
                else:
                    with open(frame, 'rb') as f:
                        raw = f.read()
                    result = remove_bg_server(prefs.server_url, raw, frame.name)
                    with open(out / (frame.stem + '_nobg.png'), 'wb') as f:
                        f.write(result)
                print(f'[BG Remover] {i+1}/{len(frames)} {frame.name}')
            except Exception as e:
                self.report({'WARNING'}, f'Skipped {frame.name}: {e}')

        self.report({'INFO'}, f'✅ Done! {len(frames)} frames saved to {out}')
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
        prefs = get_prefs(context)

        # Mode indicator
        row = layout.row()
        row.label(text='Mode:', icon='SETTINGS')
        row.label(text='Local AI' if prefs.mode == 'LOCAL' else f'Server: {prefs.server_url}')

        layout.separator()

        # Active image
        space = context.area.spaces.active if context.area else None
        if space and space.image:
            box = layout.box()
            box.label(text=space.image.name, icon='IMAGE_DATA')
            box.operator('bgremover.remove_background', icon='MATFLUID')
        else:
            layout.label(text='Open an image first', icon='INFO')

        layout.separator()

        # Render result
        if bpy.data.images.get('Render Result'):
            layout.operator('bgremover.remove_from_render', icon='RENDER_STILL')
        else:
            row = layout.row()
            row.enabled = False
            row.operator('bgremover.remove_from_render', icon='RENDER_STILL')

        layout.separator()

        # Batch sequence
        layout.operator('bgremover.process_sequence', icon='SEQUENCE')

        layout.separator()

        # Model status (local mode)
        if prefs.mode == 'LOCAL':
            if _model is not None:
                layout.label(text=f'✅ Model loaded ({_device.upper()})', icon='CHECKMARK')
            else:
                layout.label(text='Model not loaded yet', icon='TIME')
                layout.label(text='(loads on first use)', icon='BLANK1')


# ── Registration ───────────────────────────────────────────────────────────────

classes = [
    BGRemoverPreferences,
    BGRemover_OT_InstallDeps,
    BGRemover_OT_TestConnection,
    BGRemover_OT_RemoveBackground,
    BGRemover_OT_RemoveFromRender,
    BGRemover_OT_ProcessSequence,
    BGRemover_PT_Panel,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == '__main__':
    register()
