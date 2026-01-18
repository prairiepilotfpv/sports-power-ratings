"""Produce a canon coverage matrix across models/markets for a sport/season DB.

Usage: python -m src.tools.coverage_matrix --db data/db/nhl/2025-26.db --sport nhl --season 2025-26 --out outputs/coverage/nhl-2025-26-coverage.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any


MARKETS = ["ML", "SPREAD", "TOTAL"]


def extract_games_from_summary(summary: dict[str, Any]) -> int | None:
    if not isinstance(summary, dict):
        return None
    for key in ("games", "n", "count", "games_count", "games_evaluated"): 
        if key in summary:
            try:
                return int(summary[key])
            except Exception:
                continue
    # sometimes nested under metrics
    for v in summary.values():
        if isinstance(v, dict):
            for key in ("games", "n", "count"):
                if key in v:
                    try:
                        return int(v[key])
                    except Exception:
                        continue
    return None


def norm_market(m: str) -> str:
    return (m or "").strip().upper()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--sport", required=True)
    p.add_argument("--season", required=True)
    p.add_argument("--out", required=False)
    args = p.parse_args()

    db_path = Path(args.db)
    sport = args.sport
    season = args.season
    out_path = Path(args.out) if args.out else Path("outputs/coverage") / sport / f"{season}-coverage.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # gather models from several tables
    models = set()
    rows = cur.execute(
        "SELECT DISTINCT model FROM model_metrics WHERE sport=? AND season=?",
        (sport, season),
    ).fetchall()
    models.update(r[0] for r in rows if r and r[0])
    rows = cur.execute(
        "SELECT DISTINCT model FROM model_tuned_params WHERE sport=? AND season=?",
        (sport, season),
    ).fetchall()
    models.update(r[0] for r in rows if r and r[0])
    rows = cur.execute(
        "SELECT DISTINCT model FROM model_market_active_params WHERE sport=? AND season=?",
        (sport, season),
    ).fetchall()
    models.update(r[0] for r in rows if r and r[0])
    rows = cur.execute(
        "SELECT DISTINCT model FROM model_market_tuning_runs WHERE sport=? AND season=?",
        (sport, season),
    ).fetchall()
    models.update(r[0] for r in rows if r and r[0])

    models = sorted(models)

    # preload metrics per model
    metrics_by_model: dict[str, dict[str, Any] | None] = {}
    for m in models:
        row = cur.execute(
            "SELECT home_advantage, model_error, win_prob_k, base_total, margin_std, total_std, margin_mean, total_mean, backtest_log_loss, backtest_mae_margin, backtest_mae_total FROM model_metrics WHERE sport=? AND season=? AND model=?",
            (sport, season, m),
        ).fetchone()
        metrics_by_model[m] = None if row is None else {
            "home_advantage": row[0],
            "model_error": row[1],
            "win_prob_k": row[2],
            "base_total": row[3],
            "margin_std": row[4],
            "total_std": row[5],
            "margin_mean": row[6],
            "total_mean": row[7],
            "backtest_log_loss": row[8],
            "backtest_mae_margin": row[9],
            "backtest_mae_total": row[10],
        }

    # collect tuning runs summary per model+market
    tuning_runs = defaultdict(list)
    rows = cur.execute(
        "SELECT model, market, best_score, summary_metrics_json FROM model_market_tuning_runs WHERE sport=? AND season=?",
        (sport, season),
    ).fetchall()
    for model, market, best_score, summary_json in rows:
        if model is None:
            continue
        market_n = norm_market(market)
        summary = None
        if summary_json:
            try:
                summary = json.loads(summary_json)
            except Exception:
                summary = None
        tuning_runs[(model, market_n)].append({"best_score": best_score, "summary": summary})

    # active params
    active = set()
    rows = cur.execute(
        "SELECT model, market FROM model_market_active_params WHERE sport=? AND season=?",
        (sport, season),
    ).fetchall()
    for model, market in rows:
        if model and market:
            active.add((model, norm_market(market)))

    # tuned params (legacy table)
    tuned = set(r[0] for r in cur.execute("SELECT DISTINCT model FROM model_tuned_params WHERE sport=? AND season=?", (sport, season)).fetchall() if r and r[0])

    # build rows
    out_rows = []
    for m in models:
        for market in MARKETS:
            key = (m, market)
            runs = tuning_runs.get(key, [])
            tuning_exists = bool(runs)
            best_score = None
            scorable = None
            if runs:
                # pick run with smallest best_score when available, else first
                runs_with_score = [r for r in runs if r.get("best_score") is not None]
                chosen = None
                if runs_with_score:
                    chosen = sorted(runs_with_score, key=lambda x: (float(x.get("best_score") if x.get("best_score") is not None else float("inf"))))[0]
                else:
                    chosen = runs[0]
                best_score = chosen.get("best_score")
                scorable = extract_games_from_summary(chosen.get("summary") or {})

            tuned_exists = m in tuned
            active_exists = (m, market) in active
            mm = metrics_by_model.get(m)
            metric_present = False
            if mm is not None:
                if market == "ML":
                    metric_present = mm.get("win_prob_k") is not None or mm.get("backtest_log_loss") is not None
                elif market == "SPREAD":
                    metric_present = mm.get("margin_mean") is not None or mm.get("margin_std") is not None or mm.get("backtest_mae_margin") is not None
                else:  # TOTAL
                    metric_present = mm.get("total_mean") is not None or mm.get("total_std") is not None or mm.get("backtest_mae_total") is not None

            flags = []
            if not tuning_exists:
                flags.append("no_market_tuning")
            if not active_exists:
                flags.append("no_active_params")
            if not metric_present:
                flags.append("no_metrics_for_market")
            if tuned_exists:
                flags.append("has_legacy_tuned_params")

            out_rows.append({
                "model": m,
                "market": market,
                "tuning_run_exists": int(tuning_exists),
                "tuning_best_score": best_score,
                "tuned_params_exists": int(tuned_exists),
                "active_params_exists": int(active_exists),
                "metric_present": int(bool(metric_present)),
                "scorable_games": scorable,
                "flags": ";".join(flags),
            })

    # write CSV
    import csv

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["model", "market", "tuning_run_exists", "tuning_best_score", "tuned_params_exists", "active_params_exists", "metric_present", "scorable_games", "flags"],
        )
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print(f"Wrote coverage matrix to {out_path}")


if __name__ == "__main__":
    main()
