from __future__ import annotations

from pathlib import Path
import sys


def _ensure_src_on_path() -> None:
    src_dir = Path(__file__).resolve().parents[2] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def main() -> None:
    _ensure_src_on_path()
    from src.cli import pipeline

    print(
        "Warning: src.cli.run_rankings is legacy. Prefer: python -m src.cli.pipeline rank"
    )
    pipeline.main(["rank", *sys.argv[1:]])


if __name__ == "__main__":
    main()
