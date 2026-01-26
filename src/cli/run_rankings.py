from __future__ import annotations

from pathlib import Path
import sys


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    src_str = str(src_dir)
    root_str = str(repo_root)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    if root_str not in sys.path:
        insert_at = sys.path.index(src_str) + 1 if src_str in sys.path else 0
        sys.path.insert(insert_at, root_str)


def main() -> None:
    _ensure_src_on_path()
    from src.cli import pipeline

    print(
        "Warning: src.cli.run_rankings is legacy. Prefer: python -m src.cli.pipeline rank"
    )
    pipeline.main(["rank", *sys.argv[1:]])


if __name__ == "__main__":
    main()
