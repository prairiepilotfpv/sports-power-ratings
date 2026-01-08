from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Ensure project root is on sys.path so the `src` package can be imported (tests import `src.*`).
sys.path.insert(0, str(ROOT))
# Also add the inner `src` directory so modules that import top-level names (e.g., `config`) work.
sys.path.insert(0, str(ROOT / "src"))
