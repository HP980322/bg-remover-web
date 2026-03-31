# ============================================================
# bg_remover_addon.py  —  Blender Add-on
# Removes background from images/renders via local FastAPI server
#
# Installation:
#   1. Run bg_remover_server.py on your machine (port 8000)
#   2. In Blender: Edit > Preferences > Add-ons > Install
#      Select this .py file, enable it
#   3. Find "BG Remover" panel in Image Editor > N-panel > BG Remover tab
# ============================================================

bl_info = {
    "name": "BG Remover",
    "author": "HP980322",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "Image Editor > Sidebar > BG Remover",
    "description": "Remove image background via local RMBG-1.4 FastAPI server",
    "category": "Image",
}

import bpy
import io
import os
import tempfile
import urllib.request
import urllib.error
from pathlib import Path


# ── Preferences ─────────────────────────────────────────────────────────

class BGRemoverPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    server_url: bpy.props.StringProperty(
        name="Server URL",
        default="http://localhost:8000",
        description="URL of the running bg_remover_server.py",
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "server_url")
        layout.operator("bgremover.test_connection", icon="LINKED")


# ── Helpers ────────────────────────────────────────────────────────────

def get_server_url(context):
    prefs = context.preferences.addons[__name__].preferences
    return prefs.server_url.rstrip("/")


def send_image_to_server(server_url: str, img_bytes: bytes, filename: str) -> bytes:
    """POST image bytes to /remove-bg/image, return PNG bytes."""
    boundary = "----BlenderBGRemover"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + img_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{server_url}/remove-bg/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def blender_image_to_png_bytes(image: bpy.types.Image) -> bytes:
    """Save a Blender image to a temp PNG and read bytes."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        tmp = tf.name
    try:
        image.save_render(tmp)
        with open(tmp, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp)


def png_bytes_to_blender_image(png_bytes: bytes, name: str) -> bpy.types.Image:
    """Load PNG bytes as a new Blender image."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        tf.write(png_bytes)
        tmp = tf.name
    try:
        img = bpy.data.images.load(tmp)
        img.name = name
        img.pack()
        return img
    finally:
        os.unlink(tmp)


# ── Operators ────────────────────────────────────────────────────────────

class BGRemover_OT_RemoveBackground(bpy.types.Operator):
    """Remove background from the active image in the Image Editor"""
    bl_idname = "bgremover.remove_background"
    bl_label = "Remove Background"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if context.area and context.area.type == "IMAGE_EDITOR":
            return context.area.spaces.active.image is not None
        return False

    def execute(self, context):
        server_url = get_server_url(context)
        image = context.area.spaces.active.image
        self.report({"INFO"}, f"Sending '{image.name}' to server…")
        try:
            png_bytes = blender_image_to_png_bytes(image)
            result_bytes = send_image_to_server(server_url, png_bytes, image.name + ".png")
        except urllib.error.URLError as e:
            self.report({"ERROR"}, f"Cannot reach server at {server_url}: {e.reason}")
            return {"CANCELLED"}
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        out_name = Path(image.name).stem + "_nobg"
        result_img = png_bytes_to_blender_image(result_bytes, out_name)
        context.area.spaces.active.image = result_img
        self.report({"INFO"}, f"Done! Saved as '{out_name}'")
        return {"FINISHED"}


class BGRemover_OT_RemoveFromRender(bpy.types.Operator):
    """Remove background from the most recent render result"""
    bl_idname = "bgremover.remove_from_render"
    bl_label = "Remove BG from Render"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bpy.data.images.get("Render Result") is not None

    def execute(self, context):
        server_url = get_server_url(context)
        render_img = bpy.data.images["Render Result"]
        self.report({"INFO"}, "Sending render to server…")
        try:
            png_bytes = blender_image_to_png_bytes(render_img)
            result_bytes = send_image_to_server(server_url, png_bytes, "render.png")
        except urllib.error.URLError as e:
            self.report({"ERROR"}, f"Cannot reach server at {server_url}: {e.reason}")
            return {"CANCELLED"}
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        out_name = "Render_nobg"
        result_img = png_bytes_to_blender_image(result_bytes, out_name)
        for area in context.screen.areas:
            if area.type == "IMAGE_EDITOR":
                area.spaces.active.image = result_img
                break
        self.report({"INFO"}, f"Done! Saved as '{out_name}'")
        return {"FINISHED"}


class BGRemover_OT_TestConnection(bpy.types.Operator):
    """Test connection to the background removal server"""
    bl_idname = "bgremover.test_connection"
    bl_label = "Test Connection"

    def execute(self, context):
        server_url = get_server_url(context)
        try:
            req = urllib.request.Request(f"{server_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    self.report({"INFO"}, f"✅ Connected to {server_url}")
                    return {"FINISHED"}
        except Exception as e:
            self.report({"ERROR"}, f"❌ Cannot connect to {server_url}: {e}")
        return {"CANCELLED"}


# ── Panel ──────────────────────────────────────────────────────────────

class BGRemover_PT_Panel(bpy.types.Panel):
    bl_label = "BG Remover"
    bl_idname = "BGREMOVER_PT_panel"
    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category = "BG Remover"

    def draw(self, context):
        layout = self.layout
        prefs = context.preferences.addons[__name__].preferences

        col = layout.column(align=True)
        col.label(text="Server:", icon="NETWORK_DRIVE")
        col.prop(prefs, "server_url", text="")
        col.operator("bgremover.test_connection", icon="LINKED")

        layout.separator()

        space = context.area.spaces.active if context.area else None
        if space and space.image:
            box = layout.box()
            box.label(text=space.image.name, icon="IMAGE_DATA")
            box.operator("bgremover.remove_background", icon="MATFLUID")
        else:
            layout.label(text="Open an image first", icon="INFO")

        layout.separator()
        layout.operator("bgremover.remove_from_render", icon="RENDER_STILL")


# ── Registration ──────────────────────────────────────────────────────────

classes = [
    BGRemoverPreferences,
    BGRemover_OT_RemoveBackground,
    BGRemover_OT_RemoveFromRender,
    BGRemover_OT_TestConnection,
    BGRemover_PT_Panel,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
