"""Utilities for preparing script execution environments."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _REPO_ROOT / "src"


def ensure_src_on_path() -> Path:
    """Ensure the repository's ``src`` directory is importable.

    When project modules are executed as standalone scripts (``python file.py``),
    Python does not automatically include the package root on ``sys.path``. This
    helper appends the ``src`` directory exactly once so imports continue to
    work without duplicating boilerplate in every script.

    Returns
    -------
    Path
        The resolved ``src`` directory that was ensured on ``sys.path``.
    """

    src_dir = _SRC_DIR
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.append(src_str)
    return src_dir


__all__ = ["ensure_src_on_path"]
