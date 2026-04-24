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
# On first use the addon auto-installs onnxruntime==1.20.1 + numpy
# + scipy + Pillow and downloads the RMBG-1.4 ONNX model (~176 MB,
# cached forever).
# ============================================================

bl_info = {
    "name": "BG Remover",
    "author": "HP980322",
    "version": (3, 1, 5),
    "blender": (3, 0, 0),
    "location": "Image Editor > Sidebar > BG Remover",
    "description": "Remove image/render background using RMBG-1.4 AI (ONNX Runtime, web-parity cleanMask)",
    "category": "Image",
}

import bpy
import os
import sys
import json
import site
import time
import shutil
import tempfile
import threading
import importlib
import subprocess
import urllib.request
import urllib.error
from pathlib import Path


# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_URL = "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model.onnx"
MODEL_FILENAME = "rmbg_1_4.onnx"
MODEL_INPUT_SIZE = 1024

ONNXRUNTIME_SPEC = "onnxruntime==1.20.1"
REQUIRED_PACKAGES_UNPINNED = ["numpy", "scipy", "Pillow"]
VC_REDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"

_STATE_VERSION = 1


# ── Globals ────────────────────────────────────────────────────────────────────

_session = None
_session_provider = None
_last_error = None
_last_error_detail = None

_status = 'idle'
_status_message = ''
_status_progress = -1.0

_cached_problems = None
_cache_valid = False

_auto_install_attempted = False
_work_thread = None


# ── Addon Preferences ─────────────────────────────────────────────────────────

class BGRemoverPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    # User-facing name: tells users what they get, not how it works.
    # Default ON because the slower-but-cleaner result is what most want.
    use_clean_mask: bpy.props.BoolProperty(
        name="Refine edges and remove noise",
        description=(
            "Polish the AI mask after it's generated: smooth edges, drop "
            "stray pixels, and clean up sky regions. Adds about 1 second "
            "per image. Turn off only when speed matters more than quality "
            "(e.g. processing hundreds of frames)."
        ),
        default=True,
    )

    auto_install: bpy.props.BoolProperty(
        name="Auto-install dependencies on startup",
        description=(
            "If packages are missing when Blender starts, install them "
            "automatically without requiring a button click."
        ),
        default=True,
    )

    show_advanced: bpy.props.BoolProperty(
        name="Show advanced tools in panel",
        description=(
            "Show extra tools in the BG Remover panel: quality toggle, "
            "recheck, debug info, force reinstall, and clear model cache."
        ),
        default=False,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "auto_install")
        layout.prop(self, "show_advanced")

        # Quality option lives in its own labelled section.
        layout.separator()
        box = layout.box()
        box.label(text="Output quality", icon='IMAGE_DATA')
        box.prop(self, "use_clean_mask")


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


def _state_path():
    return _addon_data_dir() / 'state.json'


# ── Persistent state ──────────────────────────────────────────────────────────

def _load_state():
    try:
        p = _state_path()
        if not p.is_file():
            return {}
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if data.get('version') != _STATE_VERSION:
            return {}
        return data
    except Exception:
        return {}


def _save_state(data):
    try:
        data = dict(data)
        data['version'] = _STATE_VERSION
        with open(_state_path(), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[BG Remover] Could not save state: {e}")


def _record_healthy_state():
    _save_state({
        'healthy': True,
        'timestamp': time.time(),
        'python_executable': sys.executable,
        'onnxruntime_version': _onnxruntime_installed_version() or '',
    })


def _is_probably_still_healthy():
    state = _load_state()
    if not state.get('healthy'):
        return False
    if state.get('python_executable') != sys.executable:
        return False
    problems = _check_packages_fast()
    if problems:
        return False
    return True


# ── Status helpers ────────────────────────────────────────────────────────────

def _set_status(status, message='', progress=-1.0):
    global _status, _status_message, _status_progress
    _status = status
    _status_message = message
    _status_progress = progress
    _tag_ui_redraw()


def _tag_ui_redraw():
    try:
        wm = bpy.context.window_manager
        if wm is None:
            return
        for window in wm.windows:
            for area in window.screen.areas:
                area.tag_redraw()
    except Exception:
        pass


def _schedule_main_thread(fn, delay=0.0):
    def _wrapped():
        try:
            fn()
        except Exception as e:
            print(f"[BG Remover] main-thread callback failed: {e}")
        return None
    bpy.app.timers.register(_wrapped, first_interval=delay)


# ── Subprocess helpers ────────────────────────────────────────────────────────

def _run_no_window(cmd, **kw):
    if os.name == 'nt':
        kw.setdefault('creationflags', 0x08000000)
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

def _check_packages_fast():
    problems = []
    checks = [("onnxruntime", "onnxruntime"),
              ("numpy", "numpy"),
              ("scipy", "scipy"),
              ("Pillow", "PIL")]
    for pip_name, import_name in checks:
        try:
            importlib.import_module(import_name)
        except Exception as e:
            problems.append((pip_name, f"{type(e).__name__}: {e}"))
    return problems


def _check_packages_thorough():
    problems = []
    checks = [("onnxruntime", "onnxruntime"),
              ("numpy", "numpy"),
              ("scipy", "scipy"),
              ("Pillow", "PIL")]
    for pip_name, import_name in checks:
        if import_name in sys.modules:
            del sys.modules[import_name]
        in_process_err = None
        try:
            importlib.import_module(import_name)
        except Exception as e:
            in_process_err = f"{type(e).__name__}: {e}"

        ok, out = _probe_import_subprocess(import_name)
        if not ok:
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
    full_err = " ".join(err for _, err in problems)
    if os.name == 'nt' and (
        'dll load failed' in all_errs or
        'dynamic link library' in all_errs or
        '动态链接库' in full_err
    ):
        joined += (
            "\n\n→ Windows DLL initialization failed.\n"
            f"  1. Install Microsoft VC++ Redistributable: {VC_REDIST_URL}\n"
            "  2. Close Blender completely.\n"
            "  3. Re-open Blender and click Install Dependencies again.\n"
        )
    elif 'numpy' in all_errs and 'multiarray' in all_errs:
        joined += "\n\n→ numpy ABI mismatch. Click 'Force Reinstall onnxruntime'."
    else:
        joined += "\n\nTry: restart Blender, then click Install Dependencies again."

    return joined


def _pip_install_blocking(args):
    cmd = [sys.executable, '-m', 'pip', 'install', '--user'] + args
    try:
        result = _run_no_window(cmd, capture_output=True, text=True, timeout=600)
        ok = result.returncode == 0
        combined = (result.stdout or '') + (result.stderr or '')
        return ok, combined
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _pip_uninstall_blocking(pkg):
    try:
        _run_no_window(
            [sys.executable, '-m', 'pip', 'uninstall', '-y', pkg],
            capture_output=True, text=True, timeout=120,
        )
    except Exception:
        pass


def _install_deps_worker(force_ort=False):
    global _last_error, _last_error_detail

    def say(msg, progress=-1.0):
        _schedule_main_thread(
            lambda m=msg, p=progress: _set_status('installing', m, p)
        )

    try:
        say('Checking installed packages…')
        problems = _check_packages_thorough()
        missing = _missing_names(problems)

        installed_ort = _onnxruntime_installed_version()
        ort_needs_pin = installed_ort and installed_ort != '1.20.1'

        if not problems and not ort_needs_pin and not force_ort:
            _schedule_main_thread(lambda: _on_install_done(None))
            return

        say('Ensuring pip is available…')
        try:
            _run_no_window(
                [sys.executable, '-m', 'ensurepip', '--upgrade'],
                capture_output=True, text=True, timeout=120,
            )
        except Exception:
            pass

        if force_ort or ort_needs_pin or 'onnxruntime' in missing:
            say(f'Uninstalling existing onnxruntime '
                f'({installed_ort or "none"})…')
            _pip_uninstall_blocking('onnxruntime')

        install_list = []
        if force_ort or 'onnxruntime' in missing or ort_needs_pin:
            install_list.append(ONNXRUNTIME_SPEC)
        for pkg in REQUIRED_PACKAGES_UNPINNED:
            if pkg in missing:
                install_list.append(pkg)

        if install_list:
            say(f'Installing: {", ".join(install_list)}… (up to 2 min)')
            args = ['--upgrade'] + (['--force-reinstall', '--no-deps']
                                     if force_ort else []) + install_list
            ok, output = _pip_install_blocking(args)
            if not ok:
                raise RuntimeError(
                    f"pip install failed:\n{output[-1500:]}\n\n"
                    f"Try running Blender as Administrator."
                )

            if force_ort:
                say('Ensuring numpy is installed…')
                _pip_install_blocking(['--upgrade', 'numpy'])

        _refresh_sys_path()

        say('Verifying imports…')
        still_problems = _check_packages_thorough()
        if still_problems:
            detail = _format_install_error(still_problems)
            try:
                with open(_diagnostics_path(), 'w', encoding='utf-8') as f:
                    f.write("[BG Remover] Install succeeded but imports fail\n")
                    f.write("=" * 70 + "\n")
                    f.write(detail + "\n")
            except Exception:
                pass

            _last_error_detail = detail
            first = still_problems[0]
            err_msg = (
                f"Installed {first[0]} but can't import it. "
                f"See diagnostics.txt for details."
            )
            _schedule_main_thread(lambda m=err_msg, p=still_problems:
                                   _on_install_done(m, problems=p))
            return

        _record_healthy_state()
        _schedule_main_thread(lambda: _on_install_done(None))
    except Exception as e:
        err_str = str(e)
        _schedule_main_thread(lambda m=err_str: _on_install_done(m))


def _on_install_done(error_msg, problems=None):
    global _last_error, _cached_problems, _cache_valid, _work_thread
    _work_thread = None

    if error_msg is None:
        _last_error = None
        _cached_problems = []
        _cache_valid = True
        _set_status('idle', '✅ Ready — dependencies installed!')
        _schedule_main_thread(lambda: _set_status('idle'), delay=4.0)
    else:
        _last_error = error_msg
        if problems is not None:
            _cached_problems = problems
            _cache_valid = True
        else:
            _cached_problems = _check_packages_fast()
            _cache_valid = True
        _set_status('idle', '')


def _start_install_thread(force_ort=False):
    global _work_thread
    if _work_thread is not None and _work_thread.is_alive():
        return False
    _set_status('installing', 'Starting…')
    _work_thread = threading.Thread(
        target=_install_deps_worker,
        kwargs={'force_ort': force_ort},
        daemon=True,
    )
    _work_thread.start()
    return True


def ensure_deps_sync():
    problems = _check_packages_fast()
    if not problems:
        return True
    missing = _missing_names(problems)
    installed_ort = _onnxruntime_installed_version()
    ort_needs_pin = installed_ort and installed_ort != '1.20.1'

    try:
        _run_no_window(
            [sys.executable, '-m', 'ensurepip', '--upgrade'],
            capture_output=True, text=True, timeout=120,
        )
    except Exception:
        pass

    if ort_needs_pin or 'onnxruntime' in missing:
        _pip_uninstall_blocking('onnxruntime')

    install_list = []
    if 'onnxruntime' in missing or ort_needs_pin:
        install_list.append(ONNXRUNTIME_SPEC)
    for pkg in REQUIRED_PACKAGES_UNPINNED:
        if pkg in missing:
            install_list.append(pkg)

    if install_list:
        ok, output = _pip_install_blocking(['--upgrade'] + install_list)
        if not ok:
            raise RuntimeError(f"pip install failed:\n{output[-1500:]}")

    _refresh_sys_path()
    still = _check_packages_fast()
    if still:
        raise RuntimeError(
            f"Can't import {still[0][0]}: {still[0][1][:200]}"
        )
    _record_healthy_state()
    return True


# ── Model download ────────────────────────────────────────────────────────────

def _download_with_progress(url, dst_path, status_cb=None):
    tmp_path = dst_path.with_suffix('.part')
    print(f"[BG Remover] Downloading RMBG-1.4 ONNX model (~176 MB) from {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            total = int(resp.headers.get('Content-Length', 0))
            downloaded = 0
            last_report_pct = -1
            with open(tmp_path, 'wb') as f:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = int(100 * downloaded / total)
                        progress = downloaded / total
                        if pct != last_report_pct and pct % 2 == 0:
                            mb = downloaded / (1024 * 1024)
                            msg = f"Downloading model: {pct}% ({mb:.0f} MB)"
                            print(f"[BG Remover] {msg}")
                            if status_cb:
                                status_cb(msg, progress)
                            last_report_pct = pct
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

    ensure_deps_sync()
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


def blender_image_to_pil(image):
    ensure_deps_sync()
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
    """Install onnxruntime + numpy + scipy + Pillow in a background thread"""
    bl_idname = 'bgremover.install_deps'
    bl_label = 'Install Dependencies'

    def execute(self, context):
        global _last_error
        _last_error = None
        started = _start_install_thread(force_ort=False)
        if not started:
            self.report({'WARNING'}, 'Install already in progress.')
            return {'CANCELLED'}
        self.report({'INFO'}, 'Installing in background — see panel for progress.')
        return {'FINISHED'}


class BGRemover_OT_ForceReinstallOrt(bpy.types.Operator):
    """Force-reinstall onnxruntime 1.20.1 in a background thread"""
    bl_idname = 'bgremover.force_reinstall_ort'
    bl_label = 'Force Reinstall onnxruntime 1.20.1'

    def execute(self, context):
        global _last_error
        _last_error = None
        started = _start_install_thread(force_ort=True)
        if not started:
            self.report({'WARNING'}, 'Install already in progress.')
            return {'CANCELLED'}
        self.report({'INFO'}, 'Reinstalling in background — see panel for progress.')
        return {'FINISHED'}


class BGRemover_OT_Recheck(bpy.types.Operator):
    """Re-run the thorough package check"""
    bl_idname = 'bgremover.recheck'
    bl_label = 'Recheck Packages'

    def execute(self, context):
        _set_status('checking', 'Checking packages…')

        def worker():
            global _cached_problems, _cache_valid
            try:
                _cached_problems = _check_packages_thorough()
                _cache_valid = True
                if not _cached_problems:
                    _record_healthy_state()
            finally:
                _schedule_main_thread(lambda: _set_status('idle', ''))

        threading.Thread(target=worker, daemon=True).start()
        return {'FINISHED'}


class BGRemover_OT_Diagnostics(bpy.types.Operator):
    """Generate a detailed report of the Python environment, package
    versions, and import errors. Use this when reporting a problem."""
    bl_idname = 'bgremover.diagnostics'
    bl_label = 'Copy Debug Info for Support'

    def execute(self, context):
        import platform
        _set_status('checking', 'Running diagnostics…')

        def worker():
            lines = []

            def out(s=''):
                lines.append(s)
                print(s)

            try:
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
                out("Package import probes (subprocess — real OS loader errors):")
                for pip_name, import_name in [
                        ("onnxruntime", "onnxruntime"), ("numpy", "numpy"),
                        ("scipy", "scipy"), ("Pillow", "PIL")]:
                    ok, msg = _probe_import_subprocess(import_name)
                    tag = "OK  " if ok else "FAIL"
                    out(f"  [{tag}] {pip_name}:")
                    for line in (msg or '').splitlines()[-8:]:
                        out(f"      {line}")

                out("")
                out("pip show:")
                try:
                    result = _run_no_window(
                        [sys.executable, '-m', 'pip', 'show',
                         'onnxruntime', 'numpy', 'scipy', 'Pillow'],
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

                diag_path = _diagnostics_path()
                try:
                    with open(diag_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(lines) + '\n')
                    print(f"[BG Remover] Saved: {diag_path}")
                except Exception as e:
                    print(f"[BG Remover] Save failed: {e}")
            finally:
                _schedule_main_thread(lambda: _set_status('idle', ''))
                def refresh_worker():
                    global _cached_problems, _cache_valid
                    _cached_problems = _check_packages_thorough()
                    _cache_valid = True
                    _schedule_main_thread(_tag_ui_redraw)
                threading.Thread(target=refresh_worker, daemon=True).start()

        threading.Thread(target=worker, daemon=True).start()
        return {'FINISHED'}


class BGRemover_OT_OpenDiagnostics(bpy.types.Operator):
    """Open diagnostics.txt in the system default text editor"""
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
            ensure_deps_sync()
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

def _status_icon():
    if _status == 'installing' or _status == 'uninstalling':
        return 'SORTTIME'
    if _status == 'checking':
        return 'VIEWZOOM'
    if _status == 'downloading':
        return 'IMPORT'
    return 'BLANK1'


class BGRemover_PT_Panel(bpy.types.Panel):
    bl_label = 'BG Remover'
    bl_idname = 'BGREMOVER_PT_panel'
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'BG Remover'

    def draw(self, context):
        layout = self.layout
        prefs = get_prefs(context)

        # ── WORKING STATE — priority over everything else ───────────────────
        if _status != 'idle':
            box = layout.box()
            box.label(text='Working…', icon=_status_icon())
            if _status_message:
                msg = _status_message
                while msg:
                    chunk, msg = msg[:42], msg[42:]
                    box.label(text=chunk, icon='BLANK1')
            if _status_progress >= 0.0:
                filled = int(_status_progress * 20)
                bar = '█' * filled + '░' * (20 - filled)
                box.label(text=f"  {bar}  {int(_status_progress * 100)}%",
                          icon='BLANK1')
            box.label(text='Blender stays responsive.', icon='INFO')
            return

        if _status_message:
            box = layout.box()
            box.label(text=_status_message, icon='CHECKMARK')

        if not _cache_valid:
            box = layout.box()
            box.label(text='Checking environment…', icon='VIEWZOOM')
            return

        problems = _cached_problems or []
        missing = _missing_names(problems)
        model_exists = _model_path().is_file()

        # ── PROBLEMS ────────────────────────────────────────────────────────
        if problems:
            box = layout.box()
            box.alert = True
            box.label(text='Setup required', icon='ERROR')
            box.label(text=f"Cannot import: {', '.join(missing)}")

            first_err_line = (problems[0][1] or '').strip().splitlines()[0][:55]
            box.label(text=first_err_line, icon='INFO')

            all_errs = " ".join(err for _, err in problems).lower()
            full_err = " ".join(err for _, err in problems)
            dll_keywords = ['dll load failed', 'dynamic link library',
                            '动态链接库', '初始化例程失败']
            is_dll_init_fail = any(k in all_errs or k in full_err
                                    for k in dll_keywords)

            if os.name == 'nt' and is_dll_init_fail:
                vc_box = box.box()
                vc_box.label(text='Likely cause: missing VC++ Redist',
                             icon='INFO')
                op = vc_box.operator('bgremover.open_url',
                                     text='Install vc_redist.x64.exe',
                                     icon='URL')
                op.url = VC_REDIST_URL
                vc_box.label(text='Then close Blender, reopen, try again.',
                             icon='BLANK1')
                vc_box.operator('bgremover.force_reinstall_ort',
                                text='Reinstall onnxruntime 1.20.1',
                                icon='FILE_REFRESH')

            row = box.row(align=True)
            row.scale_y = 1.3
            row.operator('bgremover.install_deps',
                         text='Install Dependencies', icon='IMPORT')
            return

        if not model_exists:
            box = layout.box()
            box.label(text='Model not downloaded', icon='IMPORT')
            box.label(text='~176 MB, one-time', icon='INFO')
            box.operator('bgremover.install_deps',
                         text='Download Model', icon='URL')
            return

        # ── READY STATE ─────────────────────────────────────────────────────
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

        # NOTE: "Refine edges and remove noise" toggle is intentionally
        # NOT shown in the main panel — it lives in addon prefs and the
        # Advanced section. Default is on; most users never need it.

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

        # ── ADVANCED SECTION (only when enabled in prefs) ───────────────────
        if prefs and prefs.show_advanced:
            layout.separator()
            adv_box = layout.box()
            adv_box.label(text='Advanced', icon='PREFERENCES')

            # The quality toggle that used to live in the main view
            adv_box.prop(prefs, "use_clean_mask")

            row = adv_box.row(align=True)
            row.operator('bgremover.recheck', text='Recheck', icon='FILE_REFRESH')
            row.operator('bgremover.diagnostics', text='Debug Info',
                         icon='CONSOLE')
            if _diagnostics_path().is_file():
                adv_box.operator('bgremover.open_diagnostics',
                                 text='Open Debug Report', icon='TEXT')
            adv_box.operator('bgremover.force_reinstall_ort',
                             text='Reinstall onnxruntime', icon='FILE_REFRESH')
            adv_box.operator('bgremover.clear_model', icon='TRASH')


# ══════════════════════════════════════════════════════════════════════════════
# Registration
# ══════════════════════════════════════════════════════════════════════════════

classes = [
    BGRemoverPreferences,
    BGRemover_OT_InstallDeps,
    BGRemover_OT_ForceReinstallOrt,
    BGRemover_OT_Recheck,
    BGRemover_OT_Diagnostics,
    BGRemover_OT_OpenDiagnostics,
    BGRemover_OT_RemoveBackground,
    BGRemover_OT_RemoveFromRender,
    BGRemover_OT_ProcessSequence,
    BGRemover_OT_ClearModel,
    BGRemover_OT_OpenURL,
    BGRemover_PT_Panel,
]


def _startup_routine():
    global _cached_problems, _cache_valid, _auto_install_attempted

    if _is_probably_still_healthy():
        _cached_problems = []
        _cache_valid = True
        _tag_ui_redraw()
        print("[BG Remover] Healthy state cached from previous session — skipping full check.")
        return None

    def worker():
        global _cached_problems, _cache_valid, _auto_install_attempted
        try:
            problems = _check_packages_thorough()
            _cached_problems = problems
            _cache_valid = True
            if not problems:
                _record_healthy_state()
                _schedule_main_thread(_tag_ui_redraw)
                return

            prefs = get_prefs()
            should_auto = (prefs is None or prefs.auto_install) and not _auto_install_attempted
            if should_auto:
                _auto_install_attempted = True
                print("[BG Remover] Missing dependencies detected — auto-installing…")
                _schedule_main_thread(lambda: _start_install_thread(force_ort=False))
            else:
                _schedule_main_thread(_tag_ui_redraw)
        except Exception as e:
            print(f"[BG Remover] Startup check failed: {e}")
            _schedule_main_thread(_tag_ui_redraw)

    threading.Thread(target=worker, daemon=True).start()
    return None


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.app.timers.register(_startup_routine, first_interval=0.1)


def unregister():
    global _session, _session_provider, _cached_problems, _cache_valid
    global _auto_install_attempted
    _session = None
    _session_provider = None
    _cached_problems = None
    _cache_valid = False
    _auto_install_attempted = False
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == '__main__':
    register()
