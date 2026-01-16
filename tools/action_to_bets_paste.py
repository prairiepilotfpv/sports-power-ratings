"""Deprecated wrapper — use `python -m src.cli.action_paste` instead.

This file remains for backward-compatibility but exits with a message directing
users to the single supported CLI entrypoint.
"""
from __future__ import annotations

import sys


def main(argv=None):
    print("Deprecated: use 'python -m src.cli.action_paste --in <file> --out <csv>'")
    sys.exit(2)


if __name__ == "__main__":
    main()
