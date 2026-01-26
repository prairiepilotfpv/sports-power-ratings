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
    market_ocr.add_argument(
        "--captured-at",
        help=(
            "Optional ISO timestamp for captured images. If omitted, the pipeline "
            "records the current time when each image is read."
        ),
    )
    market_ocr.add_argument("--db", help="Optional DB path override")
    market_ocr.add_argument("--json-output", help="Optional JSON path to write parsed market lines (no DB writes)")

    market_commit = sub.add_parser("market-commit", help="Commit staging rows into market_snapshots")
    market_commit.add_argument("--sport", required=True)
    market_commit.add_argument("--season", required=True)
    market_commit.add_argument("--snapshot-run-id", required=True)
    market_commit.add_argument("--db", help="Optional DB path override")
    market_commit.add_argument("--require-matched", action="store_true", help="Fail if any staging rows are needs_review")
    market_commit.add_argument("--force", action="store_true", help="Force commit rows even if needs_review")

    market_csv = sub.add_parser("market-csv", help="Import market lines from a CSV into snapshots/staging")
    market_csv.add_argument("--sport", required=True)
    market_csv.add_argument("--season", required=True)
    market_csv.add_argument("--csv", dest="csv_path", required=True, help="CSV of market lines (selection, odds, line, teams)")
    market_csv.add_argument("--db", help="Optional DB path override")
    market_csv.add_argument("--default-book", dest="default_book", help="Fallback book name when the CSV omits it")
    market_csv.add_argument(
        "--date-filter",
        dest="date_filter",
        action="append",
        help="Optional game date (YYYY-MM-DD) to limit imports; can be repeated.",
    )

    clv_csv = sub.add_parser("clv-csv", help="Import closing lines/odds from a CSV and backfill bets")
    clv_csv.add_argument("--sport", required=True)
    clv_csv.add_argument("--season", required=True)
    clv_csv.add_argument("--csv", dest="csv_path", required=True, help="CSV of closing lines (selection, odds, optional game_id)")
    clv_csv.add_argument("--db", help="Optional DB path override")
    clv_csv.add_argument("--default-market-type", dest="default_market_type", help="Fallback market_type when the CSV omits it")
    clv_csv.add_argument("--captured-at", dest="captured_at", help="Override captured_at timestamp for all rows")
    clv_csv.add_argument("--no-update-bets", action="store_true", help="Skip updating existing bets with CLV values")

    parse_export = sub.add_parser("parse-export", help="Parse betting app export CSV and assign game_ids from database")
    parse_export.add_argument("--csv", dest="csv_path", required=True, help="CSV export from betting app (league, start_time, game, type, odds, stake, result)")
    parse_export.add_argument("--sport", help="Optional sport filter (nba, ncaaf, nhl, etc.); if omitted, auto-detects all sports in CSV")
    parse_export.add_argument("--season", required=True, help="Season code (e.g., 2025-26).")
    parse_export.add_argument("--db", help="Optional DB path override (use <sport> placeholder for multi-sport CSVs)")
    parse_export.add_argument("--output", help="Optional output CSV path; defaults to input filename with '-with-ids' suffix")


    import_csv = sub.add_parser("import-csv", help="Import historical bets from a CSV (e.g., from betting app export)")
    import_csv.add_argument("--csv", dest="csv_path", required=True, help="CSV of bets (league, start_time, game, type, odds, stake, result)")
    import_csv.add_argument(
        "--sport", 
        required=False, 
        help="Sport code (nba, nhl, etc.). If omitted, will infer from game_id column. CSV must contain only bets for this sport; mixed-sport CSVs will be rejected."
    )
    import_csv.add_argument("--season", required=False, help="Season code (e.g., 2025-26). If omitted, will infer from game_id column.")
    import_csv.add_argument("--db", help="Optional DB path override")
    import_csv.add_argument("--review-run-id", help="Optional review_run_id for the import batch (auto-generated if omitted)")
    import_csv.add_argument("--dry-run", action="store_true", help="Parse and validate without writing to DB")

    review_gen = sub.add_parser("review-generate", help="Generate a review workbook for a given sport/season")
    review_gen.add_argument("--sport", required=True)
    review_gen.add_argument("--season", required=True)
    review_gen.add_argument("--model", required=True)
    review_gen.add_argument("--db", help="Optional DB path override")
    review_gen.add_argument("--output-dir", help="Optional output directory")
    review_gen.add_argument("--review-run-id", help="Optional existing review_run_id to use")
    review_gen.add_argument("--snapshot-run-id", required=True, help="Market snapshot run id to evaluate")
    review_gen.add_argument("--snapshot-date", help="Filter market snapshots by captured date (YYYY-MM-DD)")
    review_gen.add_argument(
        "--formula-workbook",
        "--formula",
        dest="formula_workbook",
        action="store_true",
        help="Generate a formula-based review workbook (implied_prob/edge/ev formulas).",
    )
    review_gen.set_defaults(include_ocr_raw=True)
    ocr_group = review_gen.add_mutually_exclusive_group()
    ocr_group.add_argument(
        "--include-ocr-raw",
        dest="include_ocr_raw",
        action="store_true",
        help="Include OCR_RAW sheet with source OCR staging rows (default).",
    )
    ocr_group.add_argument(
        "--no-include-ocr-raw",
        dest="include_ocr_raw",
        action="store_false",
        help="Skip the OCR_RAW sheet in the review workbook.",
    )

    daily_workbook = sub.add_parser(
        "daily-workbook",
        help="Build a unified daily workbook (projections, snapshots, OCR, EV, BETS).",
    )
    action_import = sub.add_parser(
        "action-import",
        help="Import parser-generated CSV of market rows into snapshots/staging",
    )
    action_import.add_argument("--csv", required=True, help="CSV produced by action parser")
    action_import.add_argument("--sport", required=True)
    action_import.add_argument("--season", required=True)
    action_import.add_argument("--db", help="Optional DB path override")
    action_import.add_argument("--snapshot-run-id", help="Optional snapshot_run_id to attach to inserted snapshots")
    action_import.add_argument("--book", help="Optional book name to attach to inserted snapshots")
    daily_workbook.add_argument("--sport", required=True)
    daily_workbook.add_argument("--season", required=True)
    daily_workbook.add_argument("--date", required=True, help="Target date (YYYY-MM-DD)")
    daily_workbook.add_argument("--model", help="Optional model override for projections")
    daily_workbook.add_argument("--review-run-id", help="Optional review_run_id override")
    daily_workbook.add_argument("--snapshot-run-id", help="Optional snapshot_run_id override")
    daily_workbook.add_argument("--output", help="Optional output workbook path")
    daily_workbook.add_argument("--db", help="Optional DB path override")

    validate = sub.add_parser("validate", help="Run pre-flight validation checks")
    validate.add_argument("--sport", required=True)
    validate.add_argument("--season", required=True)
    validate.add_argument("--model", required=True)
    validate.add_argument("--date", required=True, help="Prediction date (YYYY-MM-DD)")
    validate.add_argument("--snapshot-run-id", help="Snapshot run id to validate")
    validate.add_argument("--snapshot-date", help="Snapshot date (YYYY-MM-DD) to validate")
    validate.add_argument("--min-snapshots", type=int, default=1, help="Minimum snapshot rows required (default: 1)")
    validate.add_argument("--db", help="Optional DB path override")

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
