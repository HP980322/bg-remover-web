# ============================================================
# BG Remover — Blender Add-on
# Removes backgrounds from images using RMBG-1.4 AI
#
# Two modes:
#   1. LOCAL  — runs RMBG-1.4 directly inside Blender's Python
#               (auto-installs transformers + torch on first use)
#   2. SERVER — sends images to bg_remover_server.py
#               If not running, the addon will auto-start it
#               using your system Python + ship an isolated venv.
# ============================================================

bl_info = {
    "name": "BG Remover",
    "author": "HP980322",
    "version": (2, 1, 3),
    "blender": (3, 0, 0),
    "location": "Image Editor > Sidebar > BG Remover",
    "description": "Remove image/render background using RMBG-1.4 AI (local or auto-managed server)",
    "category": "Image",
}

import bpy
import os
import io
import sys
import site
import math
import shutil
import atexit
import signal
import struct
import tempfile
import importlib
import subprocess
import urllib.request
import urllib.error
from pathlib import Path


SERVER_SCRIPT_URL = (
    "https://raw.githubusercontent.com/HP980322/bg-remover-web/"
    "main/blender/bg_remover_server.py"
)

SERVER_REQS_LIGHT = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "python-multipart>=0.0.9",
    "Pillow>=10.0",
    "numpy>=1.24,<2.0",
    "transformers>=4.40",
]

PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cpu"
SERVER_REQS_TORCH = [
    "torch>=2.0",
    "torchvision>=0.15",
]

MIN_PY = (3, 9)
MAX_PY_HINT = (3, 12)


# ── Preferences ───────────────────────────────────────────────────────────────

class BGRemoverPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    mode: bpy.props.EnumProperty(
        name="Mode",
        items=[
            ('LOCAL',  'Local (built-in AI)',  'Run RMBG-1.4 directly in Blender — no server needed'),
            ('SERVER', 'Server',               'Send images to bg_remover_server.py (auto-managed)'),
        ],
        default='LOCAL',
    )
    server_url: bpy.props.StringProperty(
        name="Server URL",
        default="http://localhost:8000",
        description="URL of bg_remover_server.py. localhost URLs are auto-started; remote URLs must be started by you.",
    )
    auto_start_server: bpy.props.BoolProperty(
        name="Auto-start server",
        default=True,
        description="When SERVER mode is on and the server URL is local, automatically launch it as a subprocess if not already running",
    )
    system_python: bpy.props.StringProperty(
        name="System Python (optional)",
        default="",
        description="Leave blank to auto-detect 'python' on PATH. Only set this if auto-detect picks the wrong version, or if Python isn't on PATH.",
        subtype='FILE_PATH',
    )
    server_script_path: bpy.props.StringProperty(
        name="Server Script (optional)",
        default="",
        description="Leave blank to auto-download bg_remover_server.py from GitHub. Only set this if you want to use a local modified copy.",
        subtype='FILE_PATH',
    )
    show_advanced: bpy.props.BoolProperty(
        name="Advanced",
        default=False,
        description="Show advanced overrides (system Python, server script path)",
    )
    model_loaded: bpy.props.BoolProperty(default=False)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "mode", expand=True)

        if self.mode == 'SERVER':
            box = layout.box()
            box.prop(self, "server_url")
            box.prop(self, "auto_start_server")
            if self.auto_start_server:
                box.label(text="First start installs ~2GB of deps — be patient.", icon='INFO')
                box.label(text="Python 3.11 or 3.12 recommended (torch wheels).", icon='INFO')

                # Advanced section — collapsed by default, these are rarely needed
                adv_header = box.row()
                adv_header.prop(
                    self, "show_advanced",
                    icon='TRIA_DOWN' if self.show_advanced else 'TRIA_RIGHT',
                    emboss=False, text="Advanced (optional overrides)",
                )
                if self.show_advanced:
                    col = box.column(align=True)
                    col.prop(self, "system_python")
                    col.prop(self, "server_script_path")
                    col.label(text="Leave both blank for auto-detect / auto-download.", icon='INFO')

            row = box.row(align=True)
            row.operator("bgremover.test_connection", icon="LINKED")
            row.operator("bgremover.start_server", icon="PLAY")
            row.operator("bgremover.stop_server", icon="PAUSE")
            row.operator("bgremover.open_log", icon="TEXT")

        row = layout.row()
        row.operator("bgremover.install_deps", icon="IMPORT")


# ── Globals ────────────────────────────────────────────────────────────────────

_model = None
_processor = None
_device = 'cpu'

_server_proc = None
_server_log_path = None
_server_workdir = None
_server_venv_python = None
_server_last_error = None


def get_prefs(context):
    return context.preferences.addons[__name__].preferences


def _addon_data_dir():
    base = Path(bpy.utils.user_resource('SCRIPTS', path='addons'))
    d = base / 'bg_remover_data'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _log_path():
    global _server_log_path
    if _server_log_path is None:
        _server_log_path = _addon_data_dir() / 'server.log'
    return _server_log_path


def _log(msg):
    print(f"[BG Remover] {msg}")
    try:
        with open(_log_path(), 'a', encoding='utf-8') as f:
            f.write(f"[addon] {msg}\n")
    except Exception:
        pass


def _run_logged(cmd, **kw):
    _log(f"$ {' '.join(str(c) for c in cmd)}")
    with open(_log_path(), 'ab', buffering=0) as log_fh:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            **kw,
        )
        rc = proc.wait()
    if rc != 0:
        tail = _read_log_tail(1500)
        raise subprocess.CalledProcessError(rc, cmd, output=tail)


def _refresh_sys_path():
    try:
        for p in site.getsitepackages() + [site.getusersitepackages()]:
            if p and p not in sys.path:
                sys.path.append(p)
        importlib.invalidate_caches()
    except Exception as e:
        _log(f"sys.path refresh warning: {e}")


def ensure_pillow():
    try:
        import PIL  # noqa: F401
        return True
    except ImportError:
        pass
    python = sys.executable
    try:
        subprocess.check_call(
            [python, '-m', 'ensurepip', '--upgrade'],
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        )
    except Exception:
        pass
    try:
        subprocess.check_call(
            [python, '-m', 'pip', 'install', '--upgrade', '--user', 'Pillow'],
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "Failed to install Pillow into Blender's Python. Try running "
            f"Blender as Administrator (Windows) / with sudo (Unix). Error: {e}"
        )
    _refresh_sys_path()
    try:
        import PIL  # noqa: F401
    except ImportError as e:
        raise RuntimeError(f"Pillow installed but not importable: {e}")
    return True


def ensure_deps():
    ensure_pillow()
    python = sys.executable
    missing = []
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append('torch')
    try:
        import transformers  # noqa: F401
    except ImportError:
        missing.append('transformers')
    try:
        import torchvision  # noqa: F401
    except ImportError:
        missing.append('torchvision')
    if missing:
        try:
            subprocess.check_call(
                [python, '-m', 'pip', 'install', '--upgrade', '--user'] + missing,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to install {', '.join(missing)} into Blender's Python. {e}"
            )
        _refresh_sys_path()
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
        _log(f"Model load failed: {e}")
        return False


# ── Inference helpers ─────────────────────────────────────────────────────────

def remove_bg_local(pil_img):
    if not load_model():
        raise RuntimeError("Could not load RMBG-1.4 model. Check console.")
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


def remove_bg_server(server_url, png_bytes, filename):
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
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def server_is_alive(server_url, timeout=2):
    try:
        req = urllib.request.Request(f"{server_url}/health", method='GET')
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _is_localhost_url(url):
    return any(h in url for h in ('localhost', '127.0.0.1', '0.0.0.0'))


# ── Server manager ────────────────────────────────────────────────────────────

def _python_version(py):
    try:
        out = subprocess.run(
            [py, '-c', 'import sys; print(sys.version_info[0], sys.version_info[1])'],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            parts = out.stdout.strip().split()
            return (int(parts[0]), int(parts[1]))
    except Exception:
        pass
    return None


def _resolve_system_python(prefs):
    candidates = []
    if prefs.system_python:
        candidates.append(prefs.system_python)
    for name in ('python3', 'python'):
        p = shutil.which(name)
        if p:
            candidates.append(p)
    if os.name == 'nt':
        for v in ('3.12', '3.11', '3.10', '3.9'):
            p = shutil.which(f'py -{v}')
            if p:
                candidates.append(p)

    blender_py = Path(sys.executable).resolve()
    for c in candidates:
        if not c:
            continue
        try:
            if Path(c).resolve() == blender_py:
                continue
        except Exception:
            pass
        ver = _python_version(c)
        if ver and ver >= MIN_PY:
            return c, ver
    return None, None


def _ensure_server_script(prefs):
    if prefs.server_script_path:
        p = Path(prefs.server_script_path)
        if p.is_file():
            return p
    here = Path(__file__).parent
    sibling = here / 'bg_remover_server.py'
    if sibling.is_file():
        return sibling
    data = _addon_data_dir()
    cached = data / 'bg_remover_server.py'
    if cached.is_file():
        return cached
    _log(f"Downloading server script from {SERVER_SCRIPT_URL}")
    try:
        with urllib.request.urlopen(SERVER_SCRIPT_URL, timeout=30) as resp:
            cached.write_bytes(resp.read())
    except Exception as e:
        raise RuntimeError(
            f"Could not download server script: {e}\n"
            f"Download {SERVER_SCRIPT_URL} manually and set 'Server Script' in preferences."
        )
    return cached


def _ensure_server_venv(system_python, sys_py_ver):
    global _server_venv_python

    venv_dir = _addon_data_dir() / 'venv'
    if os.name == 'nt':
        venv_python = venv_dir / 'Scripts' / 'python.exe'
    else:
        venv_python = venv_dir / 'bin' / 'python'

    if not venv_python.is_file():
        _log(f"Creating venv at {venv_dir}")
        _run_logged([system_python, '-m', 'venv', str(venv_dir)])

    probe = subprocess.run(
        [str(venv_python), '-c',
         'import fastapi, uvicorn, torch, transformers, torchvision, PIL, numpy; print("ok")'],
        capture_output=True, text=True,
    )
    if probe.returncode == 0:
        _server_venv_python = str(venv_python)
        return _server_venv_python

    _log("Installing server dependencies — this can take several minutes.")

    try:
        _run_logged([str(venv_python), '-m', 'pip', 'install', '--upgrade', 'pip'])
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to upgrade pip in the venv.\n"
            f"Log tail:\n{(e.output or '')[-1200:]}"
        )

    try:
        _run_logged(
            [str(venv_python), '-m', 'pip', 'install', '--upgrade'] + SERVER_REQS_LIGHT
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "pip install of fastapi/transformers/Pillow/numpy failed. "
            f"Log tail:\n{(e.output or '')[-1200:]}"
        )

    try:
        _run_logged(
            [str(venv_python), '-m', 'pip', 'install',
             '--index-url', PYTORCH_INDEX_URL] + SERVER_REQS_TORCH
        )
    except subprocess.CalledProcessError as e:
        hint = ""
        if sys_py_ver and sys_py_ver > MAX_PY_HINT:
            hint = (
                f"\n\nHINT: You're using Python {sys_py_ver[0]}.{sys_py_ver[1]}, "
                f"which often lacks PyTorch wheels. Install Python 3.11 or 3.12 "
                f"from https://www.python.org/downloads/ and set the 'System Python' "
                f"preference to its python.exe path."
            )
        raise RuntimeError(
            "pip install of torch/torchvision failed — this is the most common "
            "failure mode. Check that your system Python has available wheels.\n\n"
            f"Log tail:\n{(e.output or '')[-1200:]}"
            f"{hint}"
        )

    probe = subprocess.run(
        [str(venv_python), '-c',
         'import fastapi, uvicorn, torch, transformers, torchvision, PIL, numpy; print("ok")'],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            f"Deps installed but not all importable. stderr:\n{probe.stderr[-800:]}"
        )

    _server_venv_python = str(venv_python)
    return _server_venv_python


def _port_from_url(url):
    try:
        return int(url.rsplit(':', 1)[-1].split('/')[0])
    except Exception:
        return 8000


def start_server(prefs):
    global _server_proc, _server_workdir, _server_last_error
    _server_last_error = None

    if not _is_localhost_url(prefs.server_url):
        raise RuntimeError(
            f"Auto-start only works for local URLs. {prefs.server_url} looks remote."
        )
    if _server_proc and _server_proc.poll() is None:
        return

    system_python, sys_py_ver = _resolve_system_python(prefs)
    if not system_python:
        raise RuntimeError(
            "No system Python 3.9+ found on PATH. Install Python 3.11 or 3.12 "
            "from https://www.python.org/downloads/ (tick 'Add to PATH' during "
            "install), then restart Blender. Or set 'System Python' in addon "
            "preferences."
        )
    _log(f"Using system Python {sys_py_ver[0]}.{sys_py_ver[1]} at {system_python}")

    script = _ensure_server_script(prefs)
    venv_python = _ensure_server_venv(system_python, sys_py_ver)

    _server_workdir = script.parent
    port = _port_from_url(prefs.server_url)
    env = os.environ.copy()
    env['BG_REMOVER_PORT'] = str(port)
    env['PYTHONUNBUFFERED'] = '1'

    creationflags = 0
    if os.name == 'nt':
        creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

    log_fh = open(_log_path(), 'ab', buffering=0)
    _log(f"Launching server: {venv_python} {script} (port {port})")
    _server_proc = subprocess.Popen(
        [venv_python, str(script)],
        cwd=str(_server_workdir),
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )


def stop_server():
    global _server_proc
    if _server_proc and _server_proc.poll() is None:
        try:
            _server_proc.terminate()
            try:
                _server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _server_proc.kill()
        except Exception as e:
            _log(f"Error stopping server: {e}")
    _server_proc = None


atexit.register(stop_server)


def _read_log_tail(n=800):
    try:
        lp = _log_path()
        if lp.exists():
            with open(lp, 'rb') as f:
                f.seek(max(0, lp.stat().st_size - n * 2))
                return f.read().decode('utf-8', errors='replace')[-n:]
    except Exception:
        pass
    return ""


def ensure_server_running_modal(context, operator, on_ready):
    global _server_last_error
    prefs = get_prefs(context)

    if server_is_alive(prefs.server_url):
        on_ready()
        return True

    if not prefs.auto_start_server:
        raise RuntimeError(f"Server at {prefs.server_url} is not running and auto-start is off.")
    if not _is_localhost_url(prefs.server_url):
        raise RuntimeError(f"Cannot auto-start a remote server ({prefs.server_url}).")

    try:
        start_server(prefs)
    except Exception as e:
        _server_last_error = str(e)
        raise

    deadline = {'t': 0.0, 'max': 600.0}

    def _poll():
        global _server_last_error
        if _server_proc is None or _server_proc.poll() is not None:
            tail = _read_log_tail(1000)
            msg = f"Server process exited. See log for details.\n{tail[-500:]}"
            _server_last_error = msg
            operator.report({'ERROR'}, msg)
            return None
        if server_is_alive(prefs.server_url, timeout=1):
            try:
                on_ready()
            except Exception as e:
                operator.report({'ERROR'}, f"Post-startup call failed: {e}")
            return None
        deadline['t'] += 2.0
        if deadline['t'] >= deadline['max']:
            msg = f"Server didn't become ready within {int(deadline['max'])}s. Check {_log_path()}"
            _server_last_error = msg
            operator.report({'ERROR'}, msg)
            return None
        return 2.0

    bpy.app.timers.register(_poll, first_interval=2.0)
    operator.report(
        {'INFO'},
        "Starting server — installing deps + downloading model (~2GB). "
        "Click 'Open Log' to watch progress."
    )
    return False


# ── Blender image helpers ─────────────────────────────────────────────────────

def blender_image_to_pil(image):
    ensure_pillow()
    from PIL import Image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tf:
        tmp = tf.name
    try:
        image.save_render(tmp)
        return Image.open(tmp).copy()
    finally:
        os.unlink(tmp)


def blender_image_to_png_bytes(image):
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tf:
        tmp = tf.name
    try:
        image.save_render(tmp)
        with open(tmp, 'rb') as f:
            return f.read()
    finally:
        os.unlink(tmp)


def pil_to_blender_image(pil_img, name):
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
    """Install AI packages into Blender's Python (LOCAL mode only)"""
    bl_idname = 'bgremover.install_deps'
    bl_label = 'Install AI Dependencies'

    def execute(self, context):
        try:
            self.report({'INFO'}, 'Installing packages — this may take a few minutes…')
            ensure_deps()
            self.report({'INFO'}, '✅ Dependencies installed!')
        except Exception as e:
            self.report({'ERROR'}, f'Install failed: {e}')
            return {'CANCELLED'}
        return {'FINISHED'}


class BGRemover_OT_TestConnection(bpy.types.Operator):
    """Check the server. If it's down and auto-start is on, launch it."""
    bl_idname = 'bgremover.test_connection'
    bl_label = 'Test Connection'

    def execute(self, context):
        prefs = get_prefs(context)
        if server_is_alive(prefs.server_url):
            self.report({'INFO'}, f'✅ Connected to {prefs.server_url}')
            return {'FINISHED'}
        if prefs.auto_start_server and _is_localhost_url(prefs.server_url):
            def _done():
                print(f"[BG Remover] ✅ Server is up at {prefs.server_url}")
            try:
                ensure_server_running_modal(context, self, _done)
                return {'FINISHED'}
            except Exception as e:
                self.report({'ERROR'}, f'Auto-start failed: {e}')
                return {'CANCELLED'}
        self.report({'ERROR'}, f'❌ Cannot connect to {prefs.server_url}')
        return {'CANCELLED'}


class BGRemover_OT_StartServer(bpy.types.Operator):
    """Start the background removal server subprocess"""
    bl_idname = 'bgremover.start_server'
    bl_label = 'Start Server'

    def execute(self, context):
        prefs = get_prefs(context)
        if server_is_alive(prefs.server_url):
            self.report({'INFO'}, 'Server is already running.')
            return {'FINISHED'}
        try:
            start_server(prefs)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        self.report({'INFO'}, 'Server starting — click Open Log to watch.')
        return {'FINISHED'}


class BGRemover_OT_StopServer(bpy.types.Operator):
    """Stop the background removal server subprocess"""
    bl_idname = 'bgremover.stop_server'
    bl_label = 'Stop Server'

    def execute(self, context):
        stop_server()
        self.report({'INFO'}, 'Server stopped.')
        return {'FINISHED'}


class BGRemover_OT_OpenLog(bpy.types.Operator):
    """Open server.log in the system's default editor"""
    bl_idname = 'bgremover.open_log'
    bl_label = 'Open Server Log'

    def execute(self, context):
        log = _log_path()
        if not log.exists():
            self.report({'WARNING'}, 'No server.log yet.')
            return {'CANCELLED'}
        try:
            if os.name == 'nt':
                os.startfile(str(log))  # type: ignore[attr-defined]
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', str(log)])
            else:
                subprocess.Popen(['xdg-open', str(log)])
        except Exception as e:
            self.report({'ERROR'}, f'Could not open log: {e}')
            return {'CANCELLED'}
        return {'FINISHED'}


def _process_with_server(prefs, png_bytes, filename, out_name):
    result_bytes = remove_bg_server(prefs.server_url, png_bytes, filename)
    return png_bytes_to_blender_image(result_bytes, out_name)


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
        area = context.area
        self.report({'INFO'}, f"Processing '{image.name}'…")
        try:
            if prefs.mode == 'LOCAL':
                pil_in = blender_image_to_pil(image)
                pil_out = remove_bg_local(pil_in)
                result_img = pil_to_blender_image(pil_out, out_name)
                area.spaces.active.image = result_img
                self.report({'INFO'}, f"Done! Saved as '{out_name}'")
                return {'FINISHED'}
            png_bytes = blender_image_to_png_bytes(image)
            def _do_work():
                try:
                    result_img = _process_with_server(prefs, png_bytes, image.name + '.png', out_name)
                    area.spaces.active.image = result_img
                    print(f"[BG Remover] Done! Saved as '{out_name}'")
                except Exception as e:
                    print(f"[BG Remover] Error: {e}")
            ensure_server_running_modal(context, self, _do_work)
            return {'FINISHED'}
        except urllib.error.URLError as e:
            self.report({'ERROR'}, f"Cannot reach server: {e.reason}")
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}


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
        screen = context.screen
        self.report({'INFO'}, 'Processing render…')
        def _show_result(result_img):
            for area in screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    area.spaces.active.image = result_img
                    break
        try:
            if prefs.mode == 'LOCAL':
                pil_in = blender_image_to_pil(render_img)
                pil_out = remove_bg_local(pil_in)
                result_img = pil_to_blender_image(pil_out, out_name)
                _show_result(result_img)
                self.report({'INFO'}, f"Done! Saved as '{out_name}'")
                return {'FINISHED'}
            png_bytes = blender_image_to_png_bytes(render_img)
            def _do_work():
                try:
                    result_img = _process_with_server(prefs, png_bytes, 'render.png', out_name)
                    _show_result(result_img)
                    print(f"[BG Remover] Done! Saved as '{out_name}'")
                except Exception as e:
                    print(f"[BG Remover] Error: {e}")
            ensure_server_running_modal(context, self, _do_work)
            return {'FINISHED'}
        except urllib.error.URLError as e:
            self.report({'ERROR'}, f"Cannot reach server: {e.reason}")
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}


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
        prefs = get_prefs(context)
        src = Path(self.directory)
        out = Path(self.output_dir) if self.output_dir else src.parent / (src.name + '_nobg')
        out.mkdir(parents=True, exist_ok=True)
        frames = sorted([
            f for f in src.iterdir()
            if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.exr', '.tif')
        ])
        if not frames:
            self.report({'ERROR'}, 'No image files found.')
            return {'CANCELLED'}
        self.report({'INFO'}, f'Processing {len(frames)} frames…')
        def _run_batch():
            try:
                ensure_pillow()
                from PIL import Image
            except Exception as e:
                print(f"[BG Remover] {e}")
                return
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
                    print(f'[BG Remover] {i + 1}/{len(frames)} {frame.name}')
                except Exception as e:
                    print(f'[BG Remover] Skipped {frame.name}: {e}')
            print(f'[BG Remover] ✅ Done! {len(frames)} frames -> {out}')
        try:
            if prefs.mode == 'LOCAL':
                _run_batch()
            else:
                ensure_server_running_modal(context, self, _run_batch)
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}


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

        row = layout.row()
        row.label(text='Mode:', icon='SETTINGS')
        row.label(text='Local AI' if prefs.mode == 'LOCAL' else 'Server')

        if prefs.mode == 'SERVER':
            alive = server_is_alive(prefs.server_url, timeout=0.5)
            starting = (_server_proc is not None and _server_proc.poll() is None and not alive)
            box = layout.box()
            status_row = box.row()
            if alive:
                status_row.label(text='✅ Server running', icon='CHECKMARK')
            elif starting:
                status_row.label(text='⏳ Server starting…', icon='TIME')
            else:
                status_row.label(text='Server is not running', icon='PAUSE')

            if _server_last_error and not alive and not starting:
                err_box = box.box()
                err_box.alert = True
                first_line = _server_last_error.strip().splitlines()[0][:80]
                err_box.label(text=f"Error: {first_line}", icon='ERROR')
                err_box.operator('bgremover.open_log', text='Open Full Log', icon='TEXT')

            ctl_row = box.row(align=True)
            if alive or starting:
                ctl_row.operator('bgremover.stop_server', icon='PAUSE')
            else:
                ctl_row.operator('bgremover.start_server', icon='PLAY', text='Start Server')
            ctl_row.operator('bgremover.test_connection', icon='LINKED', text='Test')
            ctl_row.operator('bgremover.open_log', icon='TEXT', text='')

        layout.separator()

        try:
            import PIL  # noqa: F401
            pillow_ok = True
        except ImportError:
            pillow_ok = False
        if not pillow_ok:
            box = layout.box()
            box.alert = True
            box.label(text='Pillow not installed', icon='ERROR')
            box.operator('bgremover.install_deps', icon='IMPORT')
            return

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

        layout.separator()
        layout.operator('bgremover.process_sequence', icon='SEQUENCE')

        layout.separator()
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
    BGRemover_OT_StartServer,
    BGRemover_OT_StopServer,
    BGRemover_OT_OpenLog,
    BGRemover_OT_RemoveBackground,
    BGRemover_OT_RemoveFromRender,
    BGRemover_OT_ProcessSequence,
    BGRemover_PT_Panel,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    stop_server()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == '__main__':
    register()
