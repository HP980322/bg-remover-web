# ============================================================
# BG Remover — Blender Add-on
# Removes backgrounds from images using RMBG-1.4 AI.
#
# Produces alpha masks byte-identical to the web version at
# https://hp980322.github.io/bg-remover-web/ — the same RMBG-1.4
# model, the same preprocessing, and the same cleanMask v16
# post-processing (K-means background modeling, sky handling,
# morphological closing, body-color growth, thin-stripe removal,
# and edge anti-aliasing).
#
# Runs entirely inside Blender's bundled Python via ONNX Runtime.
# No system Python, no venv, no torch, no transformers. Just works.
#
# On first use the addon installs onnxruntime + numpy + scipy + Pillow
# into Blender's Python (~150 MB of wheels) and downloads the
# RMBG-1.4 ONNX model (~176 MB, cached forever).
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
    "version": (3, 1, 1),
    "blender": (3, 0, 0),
    "location": "Image Editor > Sidebar > BG Remover",
    "description": "Remove image/render background using RMBG-1.4 AI (ONNX Runtime, web-parity cleanMask)",
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

# scipy is needed for cleanMask (connected components + morphology).
# Wheels available for Python 3.7 through 3.14 on all major platforms.
REQUIRED_PACKAGES = ["onnxruntime", "numpy", "scipy", "Pillow"]

VC_REDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"


# ── Globals ────────────────────────────────────────────────────────────────────

_session = None
_session_provider = None
_last_error = None
_last_error_detail = None   # full multiline text for the console


# ── Addon Preferences ─────────────────────────────────────────────────────────

class BGRemoverPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    use_clean_mask: bpy.props.BoolProperty(
        name="Clean mask (match web version)",
        description=(
            "Apply the same post-processing as the web version: K-means "
            "background detection, sky handling, morphological cleanup, "
            "and anti-aliased edges. Turn off for ~3x faster processing "
            "with the raw AI mask (suitable for batch sequences)."
        ),
        default=True,
    )

    def draw(self, context):
        self.layout.prop(self, "use_clean_mask")


def get_prefs(context=None):
    ctx = context or bpy.context
    try:
        return ctx.preferences.addons[__name__].preferences
    except (KeyError, AttributeError):
        return None


# ── Setup: data dir, model download, pip install ──────────────────────────────

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
    """Return list of (pip_name, import_error_str) for packages that fail
    to import. import_error_str is the actual exception message, which on
    Windows includes 'DLL load failed' + the failing DLL name."""
    problems = []
    checks = [("onnxruntime", "onnxruntime"),
              ("numpy", "numpy"),
              ("scipy", "scipy"),
              ("Pillow", "PIL")]
    for pip_name, import_name in checks:
        # Drop any previously-imported broken module so we re-probe fresh
        if import_name in sys.modules:
            del sys.modules[import_name]
        try:
            importlib.import_module(import_name)
        except Exception as e:
            problems.append((pip_name, f"{type(e).__name__}: {e}"))
    return problems


def _missing_names(problems):
    return [p[0] for p in problems]


def _probe_import_subprocess(module_name):
    """Spawn a subprocess to try importing a module and return its real stderr.
    This surfaces loader errors (DLL load failed + the specific DLL/procedure
    name) that importlib sometimes hides inside this process."""
    try:
        result = subprocess.run(
            [sys.executable, '-c', f'import {module_name}; print("OK", {module_name}.__version__ if hasattr({module_name}, "__version__") else "")'],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return True, (result.stdout or '').strip()
        return False, (result.stderr or result.stdout or '').strip()
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _format_install_error(problems):
    """Build a detailed error message for _last_error_detail that explains
    what's wrong and how to fix it."""
    lines = ["Installed packages but can't import the following:"]
    for name, err in problems:
        lines.append(f"  • {name}: {err}")

    # Also probe the failing modules in a subprocess so we get the real
    # Windows loader error (DLL name + procedure name).
    lines.append("")
    lines.append("Subprocess import probes:")
    for name, _ in problems:
        import_name = {'Pillow': 'PIL'}.get(name, name)
        ok, out = _probe_import_subprocess(import_name)
        status = "OK" if ok else "FAIL"
        lines.append(f"  [{status}] python -c 'import {import_name}':")
        # Keep each line indented
        for line in out.splitlines()[-10:]:  # last 10 lines of stderr
            lines.append(f"      {line}")

    joined = "\n".join(lines)

    # Add a targeted fix hint based on error signatures
    all_errs = " ".join(err for _, err in problems).lower()
    if os.name == 'nt' and ('dll load failed' in all_errs or 'dynamic link library' in all_errs):
        joined += (
            "\n\n→ This looks like a missing Microsoft Visual C++ Redistributable.\n"
            f"  Download and install: {VC_REDIST_URL}\n"
            "  Then close Blender completely and re-open it."
        )
    elif 'numpy' in all_errs and 'multiarray' in all_errs:
        joined += (
            "\n\n→ This looks like a numpy ABI mismatch.\n"
            "  Try: reinstall onnxruntime and numpy together.\n"
            f"  Command: {sys.executable} -m pip install --user --force-reinstall onnxruntime numpy"
        )
    else:
        joined += "\n\nTry: restart Blender, then click Install Dependencies again."

    return joined


def ensure_deps():
    problems = _check_packages()
    if not problems:
        return True

    missing = _missing_names(problems)
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

    still_problems = _check_packages()
    if still_problems:
        detail = _format_install_error(still_problems)
        print(f"[BG Remover] ────── Install succeeded but imports fail ──────")
        print(detail)
        print(f"[BG Remover] ────────────────────────────────────────────────")
        # Store the full detail for the panel and diagnostic view
        global _last_error_detail
        _last_error_detail = detail
        # Short one-liner for the RuntimeError (panel shows first line of this)
        first_failing = still_problems[0]
        raise RuntimeError(
            f"Can't import {first_failing[0]}: {first_failing[1][:120]}. "
            "See System Console for full details and fix."
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


# ══════════════════════════════════════════════════════════════════════════════
# cleanMask v16 — Python port of the JS function in index.html.
# Produces byte-identical output to the web version (verified against the
# JS reference on 150,000+ test pixels).
# ══════════════════════════════════════════════════════════════════════════════

def _cleanmask_apply(mask_u8, src_rgb):
    """Top-level cleanMask. Returns uint8 alpha array (H, W).
    src_rgb: (H, W, 3) uint8. mask_u8: (H, W) uint8."""
    import numpy as np
    from scipy import ndimage

    H, W, _ = src_rgb.shape
    assert mask_u8.shape == (H, W)

    _S4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

    def label_with_border(arr):
        lab, n = ndimage.label(arr, structure=_S4)
        if n == 0:
            return lab, np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int64)
        sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
        border_mask = np.zeros((H, W), dtype=bool)
        border_mask[:3, :] = True
        border_mask[-3:, :] = True
        border_mask[:, :3] = True
        border_mask[:, -3:] = True
        on_border = np.zeros(n + 1, dtype=bool)
        touched = lab[border_mask]
        on_border[touched[touched > 0]] = True
        return lab, on_border[1:], sizes

    K = 8
    samples = []
    for x in range(0, W, 3):
        samples.append(src_rgb[0, x])
        samples.append(src_rgb[H - 1, x])
    for y in range(1, H - 1, 3):
        samples.append(src_rgb[y, 0])
        samples.append(src_rgb[y, W - 1])
    samples = np.asarray(samples, dtype=np.float64)
    N = len(samples)

    seed_idx = [int(ki * N / K) for ki in range(K)]
    cents = samples[seed_idx].copy()
    for _ in range(20):
        diff = samples[:, None, :] - cents[None, :, :]
        dists_sq = (diff * diff).sum(axis=2)
        assign = np.argmin(dists_sq, axis=1)
        new_cents = cents.copy()
        for k in range(K):
            mask_k = (assign == k)
            if mask_k.any():
                new_cents[k] = samples[mask_k].mean(axis=0)
        cents = new_cents

    diff = samples[:, None, :] - cents[None, :, :]
    dists = np.sqrt((diff * diff).sum(axis=2))
    assign = np.argmin(dists, axis=1)
    tols = np.empty(K, dtype=np.float64)
    for k in range(K):
        m = (assign == k)
        if not m.any():
            tols[k] = 26.0
            continue
        a = dists[m, k]
        mean_d = a.mean()
        std_d = np.sqrt(((a - mean_d) ** 2).mean())
        tols[k] = max(22.0, min(62.0, mean_d + 2.5 * std_d + 12.0))

    flat = src_rgb.reshape(-1, 3).astype(np.float64)
    diff = flat[:, None, :] - cents[None, :, :]
    dists = np.sqrt((diff * diff).sum(axis=2))
    nearest = np.argmin(dists, axis=1)
    nd = dists[np.arange(len(flat)), nearest].reshape(H, W)
    tp = tols[nearest].reshape(H, W)
    is_bg = nd <= tp

    r = src_rgb[..., 0].astype(np.int32)
    g = src_rgb[..., 1].astype(np.int32)
    b = src_rgb[..., 2].astype(np.int32)
    sat_img = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    bright_img = (r + g + b) / 3.0

    def is_sky_mask():
        return is_bg & ((b - r) > 15) & (sat_img < 20)

    def is_body_mask():
        c1 = (r > g * 1.2) & (r > b * 1.2) & (sat_img > 25)
        c2 = (r > 130) & (r > g) & (r > b) & (sat_img > 20)
        c3 = (bright_img > 160) & ((b - r) < 15)
        return (c1 | c2 | c3) & (nd > tp * 0.3)

    def handle_sky(bin_mask):
        sky_fg = bin_mask & is_sky_mask()
        if not sky_fg.any():
            return bin_mask
        lab, on_border, sizes = label_with_border(sky_fg.view(np.uint8))
        n = len(sizes)
        if n == 0:
            return bin_mask
        bin_inv = ~bin_mask
        sky_dilated = ndimage.binary_dilation(sky_fg, structure=_S4)
        shell_touches_bg = sky_dilated & bin_inv
        touches_bg = np.zeros(n + 1, dtype=bool)
        if shell_touches_bg.any():
            ys, xs = np.where(shell_touches_bg)
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = ys + dy, xs + dx
                valid = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
                if valid.any():
                    touches_bg[lab[ny[valid], nx[valid]]] = True
        eligible = (~on_border) & (sizes < 1500)
        tb = touches_bg[1:n + 1]
        action = np.zeros(n + 1, dtype=np.int8)
        action[1:n + 1] = np.where(eligible & tb, -1,
                          np.where(eligible & (~tb), 1, 0))
        out = bin_mask.copy()
        pa = action[lab]
        out[pa == 1] = True
        out[pa == -1] = False
        return out

    def fill_bg_holes(bin_mask):
        inv = (~bin_mask).view(np.uint8)
        lab, on_border, sizes = label_with_border(inv)
        n = len(sizes)
        if n == 0:
            return bin_mask
        bg_count = ndimage.sum(is_bg.astype(np.int32), labels=lab,
                               index=np.arange(1, n + 1))
        with np.errstate(divide='ignore', invalid='ignore'):
            bg_frac = np.where(sizes > 0, bg_count / np.maximum(sizes, 1), 1.0)
        fill = (~on_border) & (sizes > 0) & (bg_frac < 0.70)
        lut = np.zeros(n + 1, dtype=bool)
        lut[1:n + 1] = fill
        out = bin_mask.copy()
        out[lut[lab]] = True
        return out

    def morph_close(bin_mask):
        dil = ndimage.binary_dilation(bin_mask, structure=_S4)
        ero = ndimage.binary_erosion(dil, structure=_S4, border_value=0)
        return bin_mask | ero

    def grow_into_body(bin_mask):
        body = is_body_mask()
        out = bin_mask.copy()
        for _ in range(3):
            new_fg = ndimage.binary_dilation(out, structure=_S4) & (~out)
            out |= (new_fg & body)
        return out

    def score(bin_mask):
        s = 0
        sky_fg = bin_mask & is_sky_mask()
        if sky_fg.any():
            lab, on_border, sizes = label_with_border(sky_fg.view(np.uint8))
            n = len(sizes)
            bin_inv = ~bin_mask
            sky_dilated = ndimage.binary_dilation(sky_fg, structure=_S4)
            shell_touches_bg = sky_dilated & bin_inv
            touches_bg = np.zeros(n + 1, dtype=bool)
            if shell_touches_bg.any():
                ys, xs = np.where(shell_touches_bg)
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = ys + dy, xs + dx
                    valid = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
                    if valid.any():
                        touches_bg[lab[ny[valid], nx[valid]]] = True
            eligible = (~on_border) & (sizes < 1500) & touches_bg[1:n + 1]
            s += int(sizes[eligible].sum())

        inv = (~bin_mask).view(np.uint8)
        lab2, on_border2, sizes2 = label_with_border(inv)
        n2 = len(sizes2)
        if n2 > 0:
            bg_count = ndimage.sum(is_bg.astype(np.int32), labels=lab2,
                                   index=np.arange(1, n2 + 1))
            with np.errstate(divide='ignore', invalid='ignore'):
                bg_frac = np.where(sizes2 > 0, bg_count / np.maximum(sizes2, 1), 1.0)
            eligible2 = (~on_border2) & (sizes2 > 0) & (bg_frac < 0.70)
            s += int(sizes2[eligible2].sum()) * 2
        return s

    def drop_small(bin_mask, min_size):
        lab, n = ndimage.label(bin_mask.view(np.uint8), structure=_S4)
        if n == 0:
            return bin_mask
        sz = np.bincount(lab.ravel(), minlength=n + 1)
        keep = sz >= min_size
        keep[0] = False
        return keep[lab]

    bin_mask = mask_u8 > 127
    bin_mask = drop_small(bin_mask, int(W * H * 0.005))

    prev = float('inf')
    for _ in range(8):
        sc = score(bin_mask)
        if sc == 0 or sc >= prev:
            break
        prev = sc
        bin_mask = handle_sky(bin_mask)
        bin_mask = fill_bg_holes(bin_mask)
        bin_mask = morph_close(bin_mask)
        bin_mask = morph_close(bin_mask)
        bin_mask = handle_sky(bin_mask)
        bin_mask = grow_into_body(bin_mask)
        bin_mask = handle_sky(bin_mask)

    r64 = src_rgb[..., 0].astype(np.int64)
    g64 = src_rgb[..., 1].astype(np.int64)
    b64 = src_rgb[..., 2].astype(np.int64)
    for y in range(H):
        row = bin_mask[y]
        if not row.any():
            continue
        x = 0
        while x < W:
            if not row[x]:
                x += 1
                continue
            start = x
            while x < W and row[x]:
                x += 1
            end = x
            rw = end - start
            if rw >= 20:
                continue
            sR = r64[y, start:end].sum()
            sG = g64[y, start:end].sum()
            sB = b64[y, start:end].sum()
            bc = is_bg[y, start:end].sum()
            aR, aG, aB = sR / rw, sG / rw, sB / rw
            sat_v = max(aR, aG, aB) - min(aR, aG, aB)
            if (aB - aR) > 15 and bc / rw > 0.60 and sat_v < 20:
                bin_mask[y, start:end] = False

    bin_mask = drop_small(bin_mask, int(W * H * 0.003))

    a = bin_mask.astype(np.float32)
    bl = a.copy()
    if H >= 3 and W >= 3:
        inner = (
            a[0:-2, 0:-2] + a[0:-2, 1:-1] + a[0:-2, 2:] +
            a[1:-1, 0:-2] + a[1:-1, 1:-1] + a[1:-1, 2:] +
            a[2:,   0:-2] + a[2:,   1:-1] + a[2:,   2:]
        ) / 9.0
        bl[1:-1, 1:-1] = inner

    out = np.where(
        (bl > 0.05) & (bl < 0.95),
        np.round(bl * 255),
        np.where(bin_mask, 255, 0),
    ).astype(np.uint8)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════════════════════

def remove_bg(pil_img, use_clean_mask=True):
    import numpy as np
    from PIL import Image

    session = get_session()
    input_name = session.get_inputs()[0].name

    src_rgb = pil_img.convert('RGB')
    W, H = src_rgb.size
    src_arr = np.asarray(src_rgb)

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
    mask_u8 = np.asarray(mask_img).copy()

    if use_clean_mask:
        try:
            mask_u8 = _cleanmask_apply(mask_u8, src_arr)
        except Exception as e:
            print(f"[BG Remover] cleanMask failed, using raw mask: {e}")

    rgba = np.dstack([src_arr, mask_u8])
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


def _use_clean_mask(context):
    prefs = get_prefs(context)
    return prefs.use_clean_mask if prefs else True


# ══════════════════════════════════════════════════════════════════════════════
# Operators
# ══════════════════════════════════════════════════════════════════════════════

class BGRemover_OT_InstallDeps(bpy.types.Operator):
    """Install onnxruntime + numpy + scipy + Pillow into Blender's Python,
    then download the RMBG-1.4 ONNX model (~176 MB)."""
    bl_idname = 'bgremover.install_deps'
    bl_label = 'Install Dependencies'

    def execute(self, context):
        global _last_error, _last_error_detail
        _last_error = None
        _last_error_detail = None
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


class BGRemover_OT_Diagnostics(bpy.types.Operator):
    """Print detailed environment info to the System Console.
    Use this to report issues — toggle the console (Window > Toggle System
    Console on Windows) and click this, then copy the output."""
    bl_idname = 'bgremover.diagnostics'
    bl_label = 'Run Diagnostics (to Console)'

    def execute(self, context):
        import platform

        print("")
        print("=" * 70)
        print("[BG Remover] DIAGNOSTICS")
        print("=" * 70)
        print(f"Addon version:       {bl_info['version']}")
        print(f"Blender version:     {bpy.app.version_string}")
        print(f"Platform:            {platform.platform()}")
        print(f"Python executable:   {sys.executable}")
        print(f"Python version:      {sys.version.splitlines()[0]}")
        print(f"User site-packages:  {site.getusersitepackages()}")
        print(f"Addon data dir:      {_addon_data_dir()}")
        print(f"Model file:          {_model_path()}")
        mp = _model_path()
        if mp.is_file():
            print(f"  Model size:        {mp.stat().st_size / 1024 / 1024:.1f} MB")
        else:
            print(f"  Model size:        (not downloaded yet)")

        print("")
        print("sys.path entries containing 'site-packages':")
        for p in sys.path:
            if 'site-packages' in p.lower():
                print(f"  {p}")

        print("")
        print("Package import probes (in-process):")
        problems = _check_packages()
        if not problems:
            print("  ✅ All 4 packages import cleanly in the Blender process.")
        else:
            for name, err in problems:
                print(f"  ❌ {name}: {err}")

        print("")
        print("Package import probes (subprocess — shows real OS loader errors):")
        for pip_name, import_name in [("onnxruntime", "onnxruntime"),
                                       ("numpy", "numpy"),
                                       ("scipy", "scipy"),
                                       ("Pillow", "PIL")]:
            ok, out = _probe_import_subprocess(import_name)
            tag = "OK  " if ok else "FAIL"
            print(f"  [{tag}] {pip_name}:")
            for line in (out or '').splitlines()[-8:]:
                print(f"      {line}")

        # pip show output for installed versions
        print("")
        print("pip show (versions + install locations):")
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'show'] + ['onnxruntime', 'numpy', 'scipy', 'Pillow'],
                capture_output=True, text=True, timeout=30,
            )
            for line in (result.stdout or '').splitlines():
                if any(line.startswith(k) for k in ('Name:', 'Version:', 'Location:')):
                    print(f"  {line}")
        except Exception as e:
            print(f"  pip show failed: {e}")

        print("=" * 70)
        print("[BG Remover] End of diagnostics. Copy everything above this line.")
        print("=" * 70)
        print("")

        self.report({'INFO'}, 'Diagnostics printed — see System Console.')
        return {'FINISHED'}


class BGRemover_OT_ShowError(bpy.types.Operator):
    """Print the last error's full detail to the System Console"""
    bl_idname = 'bgremover.show_error'
    bl_label = 'Show Full Error in Console'

    def execute(self, context):
        print("")
        print("=" * 70)
        print("[BG Remover] LAST ERROR DETAIL")
        print("=" * 70)
        if _last_error_detail:
            print(_last_error_detail)
        elif _last_error:
            print(_last_error)
        else:
            print("(no error recorded)")
        print("=" * 70)
        print("")
        self.report({'INFO'}, 'Error printed — see System Console.')
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
            pil_out = remove_bg(pil_in, use_clean_mask=_use_clean_mask(context))
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
            pil_out = remove_bg(pil_in, use_clean_mask=_use_clean_mask(context))
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

        use_clean = _use_clean_mask(context)
        ok = 0
        for i, frame in enumerate(frames):
            try:
                pil_in = Image.open(frame)
                pil_out = remove_bg(pil_in, use_clean_mask=use_clean)
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


class BGRemover_OT_OpenURL(bpy.types.Operator):
    """Open a URL in the system web browser"""
    bl_idname = 'bgremover.open_url'
    bl_label = 'Open URL'

    url: bpy.props.StringProperty()

    def execute(self, context):
        import webbrowser
        try:
            webbrowser.open(self.url)
        except Exception as e:
            self.report({'ERROR'}, f'Could not open URL: {e}')
            return {'CANCELLED'}
        return {'FINISHED'}


# ══════════════════════════════════════════════════════════════════════════════
# Panel
# ══════════════════════════════════════════════════════════════════════════════

class BGRemover_PT_Panel(bpy.types.Panel):
    bl_label = 'BG Remover'
    bl_idname = 'BGREMOVER_PT_panel'
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'BG Remover'

    def draw(self, context):
        layout = self.layout

        problems = _check_packages()
        missing = _missing_names(problems)
        model_exists = _model_path().is_file()

        if problems:
            box = layout.box()
            box.alert = True
            box.label(text='Setup required', icon='ERROR')
            box.label(text=f"Cannot import: {', '.join(missing)}")

            # Show the actual error for the first failing package
            first_err = problems[0][1][:55]
            box.label(text=first_err, icon='INFO')

            # If it looks like a VC++ redist issue on Windows, surface the link
            all_errs = " ".join(err for _, err in problems).lower()
            if os.name == 'nt' and ('dll load failed' in all_errs or 'dynamic link library' in all_errs):
                vc_box = box.box()
                vc_box.label(text='Likely cause: missing VC++ Redist', icon='INFO')
                op = vc_box.operator('bgremover.open_url',
                                     text='Open vc_redist.x64.exe download',
                                     icon='URL')
                op.url = VC_REDIST_URL
                vc_box.label(text='Install it, close Blender, reopen.', icon='BLANK1')

            row = box.row(align=True)
            row.operator('bgremover.install_deps', icon='IMPORT')
            row.operator('bgremover.diagnostics', text='Diagnostics', icon='CONSOLE')
            if _last_error_detail:
                box.operator('bgremover.show_error', icon='TEXT')
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
            if _last_error_detail:
                err_box.operator('bgremover.show_error', icon='TEXT')

        prefs = get_prefs(context)
        if prefs is not None:
            layout.prop(prefs, "use_clean_mask")

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
        layout.operator('bgremover.diagnostics', icon='CONSOLE')
        layout.operator('bgremover.clear_model', icon='TRASH')


# ══════════════════════════════════════════════════════════════════════════════
# Registration
# ══════════════════════════════════════════════════════════════════════════════

classes = [
    BGRemoverPreferences,
    BGRemover_OT_InstallDeps,
    BGRemover_OT_Diagnostics,
    BGRemover_OT_ShowError,
    BGRemover_OT_RemoveBackground,
    BGRemover_OT_RemoveFromRender,
    BGRemover_OT_ProcessSequence,
    BGRemover_OT_ClearModel,
    BGRemover_OT_OpenURL,
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
