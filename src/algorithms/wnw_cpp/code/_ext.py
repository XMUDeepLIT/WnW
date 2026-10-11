
from __future__ import annotations

import ctypes
import os
import sys

from torch.utils.cpp_extension import load

_PLANNER = None


def _conda_lib() -> str:
    conda_prefix = os.environ.get("CONDA_PREFIX") or os.path.dirname(os.path.dirname(sys.executable))
    return os.path.join(conda_prefix, "lib")


def get_planner_module():
    global _PLANNER
    if _PLANNER is not None:
        return _PLANNER

    src = os.path.join(os.path.dirname(__file__), "planner.cpp")
    lib = _conda_lib()
    ldflags = [f"-Wl,-rpath,{lib}"] if os.path.isdir(lib) else []

    try:
        ctypes.CDLL(os.path.join(lib, "libstdc++.so.6"), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass

    try:
        _PLANNER = load(
            name="wnw_cpp_planner",
            sources=[src],
            extra_cflags=["-O3"],
            extra_ldflags=ldflags,
            verbose=False,
        )
    except ImportError as e:
        if "CXXABI" in str(e) or "libstdc++" in str(e):
            raise ImportError(
                f"{e}\n\nwnw_cpp's C++ planner needs conda's libstdc++ on the loader "
                f"path. Run with:\n    export LD_LIBRARY_PATH={lib}:$LD_LIBRARY_PATH"
            ) from e
        raise
    return _PLANNER
