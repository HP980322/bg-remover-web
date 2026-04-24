# ============================================================
# DEPRECATED — DO NOT USE
# ============================================================
#
# This file used to be the FastAPI server for the v2.x "Server mode"
# of the BG Remover Blender add-on. It hasn't been used since v3.0.0
# (October 2024), when the addon was rewritten to run inference
# directly inside Blender's bundled Python via ONNX Runtime.
#
# The current addon does NOT need a server. It does everything
# in-process. See bg_remover_addon.py.
#
# This file is kept around only so that anyone who finds an old
# copy of the v2.x README and tries to run it gets a clear message
# instead of a broken server.
#
# If you genuinely want a multi-machine setup where one box runs
# inference for many Blender clients, you could resurrect server
# mode — but it's deliberately not maintained. The single-file
# addon covers ~all real-world use.
# ============================================================

import sys

print(
    "\n"
    "================================================================\n"
    "  bg_remover_server.py is deprecated and no longer functional.\n"
    "\n"
    "  The BG Remover Blender add-on (v3.0.0 and later) runs the AI\n"
    "  model directly inside Blender's bundled Python.  No server\n"
    "  is required.\n"
    "\n"
    "  Just install bg_remover_addon.py:\n"
    "    Blender > Edit > Preferences > Add-ons > Install...\n"
    "    Pick blender/bg_remover_addon.py\n"
    "    Tick to enable.\n"
    "================================================================\n",
    file=sys.stderr,
)
sys.exit(1)
