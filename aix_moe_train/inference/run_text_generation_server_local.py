#!/usr/bin/env python3
"""Run Megatron's text generation server with repository-local imports only."""

from __future__ import annotations

import runpy
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.resolve()
LOCAL_MEGATRON_DIR = REPO_ROOT / "megatron"
TARGET_SCRIPT = REPO_ROOT / "tools" / "run_text_generation_server.py"


def bootstrap_local_megatron() -> None:
    repo_root_str = str(REPO_ROOT)
    if repo_root_str in sys.path:
        sys.path.remove(repo_root_str)
    sys.path.insert(0, repo_root_str)

    # Prevent namespace-package merging with an unrelated megatron installation
    # that may already exist in the container image.
    for module_name in list(sys.modules):
        if module_name == "megatron" or module_name.startswith("megatron."):
            del sys.modules[module_name]

    package = types.ModuleType("megatron")
    package.__file__ = str(LOCAL_MEGATRON_DIR / "__init__.py")
    package.__package__ = "megatron"
    package.__path__ = [str(LOCAL_MEGATRON_DIR)]

    spec = ModuleSpec(name="megatron", loader=None, is_package=True)
    spec.submodule_search_locations = [str(LOCAL_MEGATRON_DIR)]
    package.__spec__ = spec

    sys.modules["megatron"] = package


def main() -> None:
    if not TARGET_SCRIPT.is_file():
        raise FileNotFoundError(f"Cannot find target script: {TARGET_SCRIPT}")

    bootstrap_local_megatron()
    runpy.run_path(str(TARGET_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
