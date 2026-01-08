"""CLI commands for betting pipelines (skeleton).

Integrate with existing CLI patterns (see `src/cli/pipeline.py`). This file
will expose commands like `market-ocr`, `generate-review`, `log-bets`, and
`settle-bets`.
"""

from __future__ import annotations

import argparse


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("betting", help="Betting and market review commands")
    sub = parser.add_subparsers(dest="betting_cmd")

    market_ocr = sub.add_parser("market-ocr", help="Ingest market screenshots via OCR")
    market_ocr.add_argument("--sport", required=True)
    market_ocr.add_argument("--season", required=True)
    market_ocr.add_argument("--images", required=True, help="File or directory of images")
    market_ocr.add_argument("--book", help="Book name for ingestion")
    market_ocr.add_argument("--captured-at", help="Timestamp for captured images (ISO) if not embedded")
    market_ocr.add_argument("--db", help="Optional DB path override")
    market_ocr.add_argument("--json-output", help="Optional JSON path to write parsed market lines (no DB writes)")

    market_commit = sub.add_parser("market-commit", help="Commit staging rows into market_snapshots")
    market_commit.add_argument("--sport", required=True)
    market_commit.add_argument("--season", required=True)
    market_commit.add_argument("--snapshot-run-id", required=True)
    market_commit.add_argument("--db", help="Optional DB path override")
    market_commit.add_argument("--require-matched", action="store_true", help="Fail if any staging rows are needs_review")
    market_commit.add_argument("--force", action="store_true", help="Force commit rows even if needs_review")

    review_gen = sub.add_parser("review-generate", help="Generate a review workbook for a given sport/season")
    review_gen.add_argument("--sport", required=True)
    review_gen.add_argument("--season", required=True)
    review_gen.add_argument("--model", required=True)
    review_gen.add_argument("--db", help="Optional DB path override")
    review_gen.add_argument("--output-dir", help="Optional output directory")
    review_gen.add_argument("--review-run-id", help="Optional existing review_run_id to use")

    log_bets = sub.add_parser("log-bets", help="Log bets from a workbook into the DB")
    log_bets.add_argument("--workbook", required=True)
    log_bets.add_argument("--db", help="Optional DB path override")
    log_bets.add_argument("--dry-run", action="store_true")
    log_bets.add_argument("--writeback", action="store_true", help="Write back bet_id/logged_at to workbook if provided")

    settle = sub.add_parser("settle-bets", help="Settle bets for a sport/season")
    settle.add_argument("--sport", required=True)
    settle.add_argument("--season", required=True)
    settle.add_argument("--db", help="Optional DB path override")
    settle.add_argument("--settle-date", help="Optional settle date (YYYY-MM-DD)")

    report = sub.add_parser("report", help="Aggregated reports of bets")
    report.add_argument("--sport", required=True)
    report.add_argument("--season", required=True)
    report.add_argument("--db", help="Optional DB path override")
    report.add_argument(
        "--type",
        dest="report_type",
        choices=["daily", "weekly", "monthly"],
        default="daily",
        help="Report period (default: daily)",
    )
    report.add_argument("--start", help="Optional start date (YYYY-MM-DD) for the report")
    report.add_argument("--end", help="Optional end date (YYYY-MM-DD) for the report")
    report.add_argument(
        "--format",
        choices=["csv", "xlsx"],
        help="Output format (overrides --output extension)",
    )
    report.add_argument("--output", help="Output CSV/Excel path")


__all__ = ["add_subparser"]
