# ============================================================
# BG Remover — Blender Add-on
# Removes backgrounds from images using RMBG-1.4 AI.
#
# Produces alpha masks byte-identical to the web version at
# https://hp980322.github.io/bg-remover-web/ — the same RMBG-1.4
# model, the same preprocessing, and the same cleanMask v16
# post-processing.
#
# Runs entirely inside Blender's bundled Python via ONNX Runtime.
# No system Python, no venv, no torch, no transformers. Just works.
#
# On first use the addon installs onnxruntime==1.20.1 + numpy + scipy
# + Pillow into Blender's Python and downloads the RMBG-1.4 ONNX
# model (~176 MB, cached forever).
# ============================================================

bl_info = {
    "name": "BG Remover",
    "author": "HP980322",
    "version": (3, 1, 2),
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

# onnxruntime pinned to 1.20.1 — the last version before the DLL-loading
# regression that appeared in 1.22.x (and lingering issues in 1.21.x).
# See microsoft/onnxruntime issues #24907, #21270: 1.22+ no longer
# searches PATH for runtime DLLs and fails to init with
# "A dynamic link library (DLL) initialization routine failed"
# even with the latest VC++ redist installed.
ONNXRUNTIME_SPEC = "onnxruntime==1.20.1"

# scipy/numpy/Pillow don't need pinning — any recent version works.
REQUIRED_PACKAGES_UNPINNED = ["numpy", "scipy", "Pillow"]

VC_REDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"


# ── Globals ────────────────────────────────────────────────────────────────────

_session = None
_session_provider = None
_last_error = None
_last_error_detail = None


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


# ── Paths ─────────────────────────────────────────────────────────────────────

def _addon_data_dir():
    base = Path(bpy.utils.user_resource('SCRIPTS', path='addons'))
    d = base / 'bg_remover_data'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _model_path():
    return _addon_data_dir() / MODEL_FILENAME


def _diagnostics_path():
    return _addon_data_dir() / 'diagnostics.txt'


# ── Subprocess helpers ────────────────────────────────────────────────────────

def _run_no_window(cmd, **kw):
    """subprocess.run with CREATE_NO_WINDOW on Windows so pip doesn't flash
    a console window on every call."""
    if os.name == 'nt':
        kw.setdefault('creationflags', 0x08000000)  # CREATE_NO_WINDOW
    return subprocess.run(cmd, **kw)


def _refresh_sys_path():
    try:
        for p in site.getsitepackages() + [site.getusersitepackages()]:
            if p and p not in sys.path:
                sys.path.append(p)
        importlib.invalidate_caches()
    except Exception as e:
        print(f"[BG Remover] sys.path refresh warning: {e}")


def _probe_import_subprocess(module_name):
    """Spawn a subprocess to try importing a module. Returns (ok, stderr).
    This surfaces loader errors (DLL load failed + DLL name) that importlib
    sometimes hides inside this process."""
    try:
        result = _run_no_window(
            [sys.executable, '-c',
             f'import {module_name}; '
             f'print("OK", {module_name}.__version__ '
             f'if hasattr({module_name}, "__version__") else "")'],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return True, (result.stdout or '').strip()
        return False, (result.stderr or result.stdout or '').strip()
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _onnxruntime_installed_version():
    """Return installed onnxruntime version via pip show, or None."""
    try:
        result = _run_no_window(
            [sys.executable, '-m', 'pip', 'show', 'onnxruntime'],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return None
        for line in (result.stdout or '').splitlines():
            if line.startswith('Version:'):
                return line.split(':', 1)[1].strip()
    except Exception:
        pass
    return None


# ── Package checking ──────────────────────────────────────────────────────────

def _check_packages():
    """Return list of (pip_name, import_error_str). Each entry is a package
    that either can't be imported in-process OR can't be imported via
    subprocess (which catches DLL init failures that importlib silently
    swallows)."""
    problems = []
    checks = [("onnxruntime", "onnxruntime"),
              ("numpy", "numpy"),
              ("scipy", "scipy"),
              ("Pillow", "PIL")]
    for pip_name, import_name in checks:
        # Step 1: in-process import
        if import_name in sys.modules:
            del sys.modules[import_name]
        in_process_err = None
        try:
            importlib.import_module(import_name)
        except Exception as e:
            in_process_err = f"{type(e).__name__}: {e}"

        # Step 2: subprocess probe (catches DLL init failures the in-process
        # importer may miss or reformat). If subprocess fails, we trust that.
        ok, out = _probe_import_subprocess(import_name)
        if not ok:
            # Prefer the subprocess error — it's the authoritative one from
            # the OS loader.
            problems.append((pip_name, (out or in_process_err or 'unknown').strip()))
        elif in_process_err is not None:
            problems.append((pip_name, in_process_err))
    return problems


def _missing_names(problems):
    return [p[0] for p in problems]


# ── Install logic ─────────────────────────────────────────────────────────────

def _format_install_error(problems):
    lines = ["Installed packages but can't import the following:"]
    for name, err in problems:
        lines.append(f"  • {name}:")
        for subline in (err or '').splitlines()[:6]:
            lines.append(f"      {subline}")

    joined = "\n".join(lines)

    all_errs = " ".join(err for _, err in problems).lower()
    # Windows DLL init failure — most common, show targeted fix
    if os.name == 'nt' and (
        'dll load failed' in all_errs or
        'dynamic link library' in all_errs or
        '动态链接库' in " ".join(err for _, err in problems)
    ):
        joined += (
            "\n\n→ Windows DLL initialization failed.\n"
            f"  1. Install Microsoft VC++ Redistributable: {VC_REDIST_URL}\n"
            "  2. Close Blender completely (quit the program entirely).\n"
            "  3. Re-open Blender and click Install Dependencies again.\n"
            "\n"
            "  If the error persists after installing VC++ redist, this is\n"
            "  likely a known onnxruntime regression (issue #24907 on\n"
            "  microsoft/onnxruntime). The addon already pins onnxruntime\n"
            "  to the last known-good version (1.20.1) but if a newer\n"
            "  version got installed, click Install Dependencies again to\n"
            "  force the pin."
        )
    elif 'numpy' in all_errs and 'multiarray' in all_errs:
        joined += (
            "\n\n→ numpy ABI mismatch. Try:\n"
            f"  {sys.executable} -m pip install --user --force-reinstall "
            f"{ONNXRUNTIME_SPEC} numpy"
        )
    else:
        joined += "\n\nTry: restart Blender, then click Install Dependencies again."

    return joined


def _pip_install(args):
    """Run pip install with given args. Returns (ok, combined_output)."""
    cmd = [sys.executable, '-m', 'pip', 'install', '--user'] + args
    try:
        result = _run_no_window(cmd, capture_output=True, text=True, timeout=600)
        ok = result.returncode == 0
        combined = (result.stdout or '') + (result.stderr or '')
        return ok, combined
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _pip_uninstall(pkg):
    try:
        _run_no_window(
            [sys.executable, '-m', 'pip', 'uninstall', '-y', pkg],
            capture_output=True, text=True, timeout=120,
        )
    except Exception:
        pass


def ensure_deps():
    """Install missing packages into Blender's Python. Handles the onnxruntime
    version pin specifically: if a different version is already installed,
    uninstall it before reinstalling."""
    global _last_error_detail

    problems = _check_packages()
    missing_names_initial = _missing_names(problems)

    # If onnxruntime is installed but at the wrong version, force-swap it
    installed_ort = _onnxruntime_installed_version()
    ort_needs_pin = installed_ort and installed_ort != '1.20.1'

    if not problems and not ort_needs_pin:
        return True

    python = sys.executable

    try:
        subprocess.check_call(
            [python, '-m', 'ensurepip', '--upgrade'],
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        )
    except Exception:
        pass

    # If onnxruntime is installed but wrong version, or failed to import,
    # uninstall first so pip doesn't treat the install as already-satisfied.
    if ort_needs_pin or any(n == 'onnxruntime' for n in missing_names_initial):
        print(f"[BG Remover] Removing existing onnxruntime "
              f"(version {installed_ort or '?'}) before reinstalling pinned 1.20.1")
        _pip_uninstall('onnxruntime')

    # Build install list
    install_list = []
    if any(n == 'onnxruntime' for n in missing_names_initial) or ort_needs_pin:
        install_list.append(ONNXRUNTIME_SPEC)
    for pkg in REQUIRED_PACKAGES_UNPINNED:
        if pkg in missing_names_initial:
            install_list.append(pkg)

    if install_list:
        print(f"[BG Remover] Installing: {', '.join(install_list)}")
        ok, output = _pip_install(['--upgrade'] + install_list)
        if not ok:
            raise RuntimeError(
                f"pip install failed:\n{output[-1500:]}\n\n"
                f"Try running Blender as Administrator (Windows), or install "
                f"manually:\n  {python} -m pip install --user "
                f"{' '.join(install_list)}"
            )

    _refresh_sys_path()

    still_problems = _check_packages()
    if still_problems:
        detail = _format_install_error(still_problems)
        # Persist to file so we don't lose it to console scrollback
        try:
            with open(_diagnostics_path(), 'w', encoding='utf-8') as f:
                f.write("[BG Remover] Install succeeded but imports fail\n")
                f.write("=" * 70 + "\n")
                f.write(detail + "\n")
        except Exception:
            pass

        print(f"[BG Remover] ────── Install succeeded but imports fail ──────")
        print(detail)
        print(f"[BG Remover] ──────────────────────────────────────────────────")
        print(f"[BG Remover] Detail also saved to: {_diagnostics_path()}")

        _last_error_detail = detail
        first_failing = still_problems[0]
        raise RuntimeError(
            f"Can't import {first_failing[0]}. "
            f"See {_diagnostics_path()} for details."
        )
    return True


# ── Model download ────────────────────────────────────────────────────────────

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
    """Install onnxruntime==1.20.1 + numpy + scipy + Pillow into Blender's
    Python, then download the RMBG-1.4 ONNX model (~176 MB)."""
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
    """Write environment info to disk and the console. The saved file
    can be opened from the panel to copy-paste its contents."""
    bl_idname = 'bgremover.diagnostics'
    bl_label = 'Run Diagnostics'

    def execute(self, context):
        import platform

        lines = []

        def out(s=''):
            lines.append(s)
            print(s)

        out("=" * 70)
        out("[BG Remover] DIAGNOSTICS")
        out("=" * 70)
        out(f"Addon version:       {bl_info['version']}")
        out(f"Blender version:     {bpy.app.version_string}")
        out(f"Platform:            {platform.platform()}")
        out(f"Python executable:   {sys.executable}")
        out(f"Python version:      {sys.version.splitlines()[0]}")
        out(f"User site-packages:  {site.getusersitepackages()}")
        out(f"Addon data dir:      {_addon_data_dir()}")
        out(f"Model file:          {_model_path()}")
        mp = _model_path()
        if mp.is_file():
            out(f"  Model size:        {mp.stat().st_size / 1024 / 1024:.1f} MB")
        else:
            out(f"  Model size:        (not downloaded yet)")

        out("")
        out("sys.path entries with 'site-packages':")
        for p in sys.path:
            if 'site-packages' in p.lower():
                out(f"  {p}")

        out("")
        out("Package import probes (subprocess — shows real OS loader errors):")
        for pip_name, import_name in [("onnxruntime", "onnxruntime"),
                                       ("numpy", "numpy"),
                                       ("scipy", "scipy"),
                                       ("Pillow", "PIL")]:
            ok, msg = _probe_import_subprocess(import_name)
            tag = "OK  " if ok else "FAIL"
            out(f"  [{tag}] {pip_name}:")
            for line in (msg or '').splitlines()[-8:]:
                out(f"      {line}")

        out("")
        out("pip show (versions + install locations):")
        try:
            result = _run_no_window(
                [sys.executable, '-m', 'pip', 'show', 'onnxruntime', 'numpy', 'scipy', 'Pillow'],
                capture_output=True, text=True, timeout=30,
            )
            for line in (result.stdout or '').splitlines():
                if any(line.startswith(k) for k in ('Name:', 'Version:', 'Location:')):
                    out(f"  {line}")
                elif line.strip() == '---':
                    out(f"  ---")
        except Exception as e:
            out(f"  pip show failed: {e}")

        out("=" * 70)
        out("[BG Remover] End of diagnostics.")
        out("=" * 70)

        # Save to disk
        diag_path = _diagnostics_path()
        try:
            with open(diag_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
            self.report({'INFO'}, f'Diagnostics saved: {diag_path}')
            print(f"[BG Remover] Diagnostics also saved to: {diag_path}")
        except Exception as e:
            self.report({'ERROR'}, f'Could not save diagnostics: {e}')
        return {'FINISHED'}


class BGRemover_OT_OpenDiagnostics(bpy.types.Operator):
    """Open the diagnostics.txt file in the system default text editor"""
    bl_idname = 'bgremover.open_diagnostics'
    bl_label = 'Open Diagnostics File'

    def execute(self, context):
        diag_path = _diagnostics_path()
        if not diag_path.is_file():
            self.report({'WARNING'}, 'No diagnostics saved yet — run Diagnostics first.')
            return {'CANCELLED'}
        try:
            if os.name == 'nt':
                os.startfile(str(diag_path))
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', str(diag_path)])
            else:
                subprocess.Popen(['xdg-open', str(diag_path)])
        except Exception as e:
            self.report({'ERROR'}, f'Could not open file: {e}')
            return {'CANCELLED'}
        return {'FINISHED'}


class BGRemover_OT_ForceReinstallOrt(bpy.types.Operator):
    """Force-reinstall onnxruntime at the pinned version (1.20.1).
    Use this if the install seems broken."""
    bl_idname = 'bgremover.force_reinstall_ort'
    bl_label = 'Force Reinstall onnxruntime 1.20.1'

    def execute(self, context):
        global _last_error, _last_error_detail
        _last_error = None
        _last_error_detail = None
        try:
            self.report({'INFO'}, 'Uninstalling onnxruntime…')
            _pip_uninstall('onnxruntime')
            self.report({'INFO'}, 'Installing onnxruntime==1.20.1…')
            ok, output = _pip_install(['--upgrade', '--force-reinstall', '--no-deps', ONNXRUNTIME_SPEC])
            if not ok:
                raise RuntimeError(f"Install failed:\n{output[-1500:]}")
            # Also make sure its numpy dependency is there
            _pip_install(['--upgrade', 'numpy'])
            _refresh_sys_path()
            ok, out = _probe_import_subprocess('onnxruntime')
            if not ok:
                raise RuntimeError(f"Still can't import onnxruntime:\n{out}")
            self.report({'INFO'}, f'✅ onnxruntime reinstalled: {out}')
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

            # Show the first-line of the first failing package's actual error
            first_err_line = (problems[0][1] or '').strip().splitlines()[0][:55]
            box.label(text=first_err_line, icon='INFO')

            all_errs = " ".join(err for _, err in problems).lower()
            # Detect DLL-init failure across locales (Chinese Windows returns
            # the error message translated)
            full_err = " ".join(err for _, err in problems)
            dll_keywords = ['dll load failed', 'dynamic link library',
                            '动态链接库', '初始化例程失败']
            is_dll_init_fail = any(k in all_errs or k in full_err for k in dll_keywords)

            if os.name == 'nt' and is_dll_init_fail:
                vc_box = box.box()
                vc_box.label(text='Likely cause: missing VC++ Redist', icon='INFO')
                op = vc_box.operator('bgremover.open_url',
                                     text='Install vc_redist.x64.exe',
                                     icon='URL')
                op.url = VC_REDIST_URL
                vc_box.label(text='Then close Blender, reopen, try again.',
                             icon='BLANK1')
                # Offer force-reinstall as an alternative
                vc_box.operator('bgremover.force_reinstall_ort',
                                text='Reinstall onnxruntime 1.20.1',
                                icon='FILE_REFRESH')

            row = box.row(align=True)
            row.operator('bgremover.install_deps', icon='IMPORT')
            row.operator('bgremover.diagnostics', text='Diagnostics', icon='CONSOLE')
            if _diagnostics_path().is_file():
                box.operator('bgremover.open_diagnostics',
                             text='Open diagnostics.txt', icon='TEXT')
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
            if _diagnostics_path().is_file():
                err_box.operator('bgremover.open_diagnostics',
                                 text='Open diagnostics.txt', icon='TEXT')

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
    BGRemover_OT_OpenDiagnostics,
    BGRemover_OT_ForceReinstallOrt,
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
