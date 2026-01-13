from __future__ import annotations

from pathlib import Path
import sys


def _ensure_src_on_path() -> None:
    src_dir = Path(__file__).resolve().parents[2] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _build_pipeline_args(argv: list[str]) -> list[str]:
    args = list(argv)
    input_path = None
    for idx, value in enumerate(args):
        if value.startswith("-"):
            continue
        input_path = value
        args.pop(idx)
        break
    command = ["import"]
    if input_path:
        command.extend(["--input", input_path])
    return command + args


def main() -> None:
    _ensure_src_on_path()
    from src.cli import pipeline

    print(
        "Warning: src.cli.ingest is legacy. Prefer: python -m src.cli.pipeline import"
    )
    pipeline.main(_build_pipeline_args(sys.argv[1:]))


if __name__ == "__main__":
    main()
