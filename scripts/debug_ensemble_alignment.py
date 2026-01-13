"""Diagnostic helper: inspect per-model game_key sets used during ensemble tuning.

Run this with the same args you pass to `tune-ensemble` to see why predictions
don't overlap across models.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repo root is on sys.path so `src` imports work when running script directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Also add the `src` package directory so imports like `backtest` resolve.
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd

from src.pipelines import ensemble_tuning as et
from models.registry import get_backtest_model


def summarize(df: pd.DataFrame, n_samples: int = 5) -> str:
    if df is None or df.empty:
        return "(no rows)"
    keys = df["game_key"].astype(str).unique().tolist()
    return f"count={len(keys)}, sample={keys[:n_samples]}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Debug ensemble alignment")
    parser.add_argument("--sport", required=True)
    parser.add_argument("--season", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--market", required=False, default="ML", help="ML, spread, or total")
    parser.add_argument("--csv", dest="csv_path", required=False)
    parser.add_argument("--db", dest="db_path", required=False)
    parser.add_argument("--models", nargs="*", help="optional model list")
    args = parser.parse_args(argv)

    model_list = et._resolve_ml_models(args.models) if args.models else et._resolve_ml_models(None)
    print(f"Models to inspect: {model_list}")

    games_df = et._load_games_df(args.sport, args.season, csv_path=args.csv_path, db_path=args.db_path)
    print(f"Loaded games: {len(games_df)} rows")

    preds_by_model = {}
    # determine market enum
    market_str = (args.market or "ML").strip().upper()
    if market_str == "ML":
        market = et.Market.ML
    elif market_str == "SPREAD":
        market = et.Market.SPREAD
    else:
        market = et.Market.TOTAL

    for model_name in model_list:
        print(f"\nRunning backtest for model: {model_name}")
        model_cls = get_backtest_model(model_name)
        params = et._load_market_model_params(db_path=args.db_path, sport=args.sport, season=args.season, model=model_name, market=market)
        try:
            from backtest.runner import run_backtest

            outputs = run_backtest(
                model_factory=et._build_model_factory(model_cls, params),
                games_df=games_df,
                start_date=args.start_date,
                end_date=args.end_date,
                model_name=model_name,
            )
            preds = et._prepare_predictions_for_market(outputs.predictions, model_name=model_name, market=market)
            preds_by_model[model_name] = preds
            print(f"Predictions: {summarize(preds)}")
            print(preds.head(3).to_string(index=False))
        except Exception as e:
            print(f"Error running backtest for {model_name}: {e}")

    sets = {m: set(df['game_key'].astype(str).tolist()) for m, df in preds_by_model.items() if df is not None and not df.empty}
    if not sets:
        print("No prediction sets available to compare.")
        return 0

    names = list(sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = names[i]
            b = names[j]
            inter = sets[a] & sets[b]
            print(f"Intersection {a} ∩ {b}: {len(inter)} games")
            if len(inter) > 0:
                sample = list(inter)[:5]
                print(f" Sample keys: {sample}")

    common = set.intersection(*[s for s in sets.values()]) if sets else set()
    print(f"\nGlobal intersection across all models: {len(common)} games")
    if common:
        print(list(common)[:5])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
