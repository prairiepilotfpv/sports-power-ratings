"""
Simple ensemble weight optimizer.
- Accepts either a directory of per-model CSVs (each file named <model>.csv with columns: game_id,prob)
  or a single CSV with columns: game_id,<model1>,<model2>,...
- Labels CSV must contain: game_id,home_score,away_score (binary outcome: home_win)
- Optimizes weights constrained to the simplex (weights >= 0, sum = 1) using projected gradient descent
- Writes weights JSON to output path

Designed to be dependency-light: uses numpy and pandas only.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def load_preds_from_dir(preds_dir: Path) -> Tuple[np.ndarray, List[str], np.ndarray]:
    files = sorted([p for p in preds_dir.glob("*.csv")])
    if not files:
        raise FileNotFoundError(f"No CSVs found in {preds_dir}")
    dfs = []
    model_names = []
    for p in files:
        model_name = p.stem
        df = pd.read_csv(p)
        if 'game_id' not in df.columns or 'prob' not in df.columns:
            raise ValueError(f"{p} must contain 'game_id' and 'prob' columns")
        dfs.append(df[['game_id','prob']].rename(columns={'prob': model_name}))
        model_names.append(model_name)
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on='game_id', how='inner')
    game_ids = merged['game_id'].values
    probs = merged[model_names].to_numpy(dtype=float)
    return probs, model_names, game_ids


def load_preds_from_csv(preds_csv: Path) -> Tuple[np.ndarray, List[str], np.ndarray]:
    df = pd.read_csv(preds_csv)
    if 'game_id' not in df.columns:
        raise ValueError('preds CSV must contain game_id column')
    model_names = [c for c in df.columns if c != 'game_id']
    probs = df[model_names].to_numpy(dtype=float)
    game_ids = df['game_id'].values
    return probs, model_names, game_ids


def load_labels(labels_csv: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(labels_csv)
    if 'game_id' not in df.columns:
        raise ValueError('labels CSV must contain game_id column')
    if not ({'home_score','away_score'} <= set(df.columns)):
        raise ValueError("labels CSV must contain 'home_score' and 'away_score'")
    y = (df['home_score'] > df['away_score']).astype(int).values
    game_ids = df['game_id'].values
    return y, game_ids


def align_preds_labels(pred_game_ids: np.ndarray, y_game_ids: np.ndarray, probs: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Align based on game_id intersection
    pred_idx = {gid: i for i, gid in enumerate(pred_game_ids)}
    y_idx = {gid: i for i, gid in enumerate(y_game_ids)}
    common = [gid for gid in pred_game_ids if gid in y_idx]
    if not common:
        raise ValueError('No overlapping game_id between preds and labels')
    p_inds = [pred_idx[g] for g in common]
    y_inds = [y_idx[g] for g in common]
    aligned_probs = probs[p_inds, :]
    aligned_y = y[y_inds]
    return aligned_probs, aligned_y


def log_loss(y_true: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    # From: Efficient Projections onto the l1-Ball for Learning in High Dimensions
    # and projection onto simplex algorithm
    n = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0]
    if rho.size == 0:
        theta = 0.0
    else:
        rho = rho[-1]
        theta = (cssv[rho] - 1) / (rho + 1)
    w = np.maximum(v - theta, 0)
    return w


def optimize_weights(probs: np.ndarray,
                     y: np.ndarray,
                     lr: float = 0.2,
                     max_iter: int = 2000,
                     tol: float = 1e-6,
                     restarts: int = 3,
                     verbose: bool = False) -> Tuple[np.ndarray, float]:
    n_models = probs.shape[1]
    best_w = None
    best_loss = np.inf
    N = probs.shape[0]
    for r in range(restarts):
        w = np.random.rand(n_models)
        w = w / w.sum()
        prev_loss = np.inf
        for it in range(max_iter):
            ensemble = probs.dot(w)
            loss = log_loss(y, ensemble)
            if verbose and (it % 200 == 0):
                print(f"restart={r} iter={it} loss={loss:.6f}")
            if prev_loss - loss < 0 and it > 5:
                # small backtracking on step size
                lr *= 0.5
            prev_loss = loss
            # gradient calculation
            denom = ensemble * (1 - ensemble)
            # avoid division issues
            denom = np.clip(denom, 1e-8, None)
            residual = (ensemble - y) / denom  # shape (N,)
            grad = (residual[:, None] * probs).mean(axis=0)  # shape (n_models,)
            w = w - lr * grad
            w = project_to_simplex(w)
            if it % 50 == 0:
                # check convergence every 50 iters
                if abs(loss - prev_loss) < tol:
                    break
        final_loss = log_loss(y, probs.dot(w))
        if final_loss < best_loss:
            best_loss = final_loss
            best_w = w.copy()
    return best_w, best_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preds-dir', type=str, help='Directory containing <model>.csv files with columns game_id,prob')
    p.add_argument('--preds-csv', type=str, help='Single CSV with columns: game_id,model1,model2,...')
    p.add_argument('--labels', type=str, required=True, help='Labels CSV with game_id,home_score,away_score')
    p.add_argument('--out', type=str, default='outputs/tuning/ensembles/weights.json', help='Output JSON path')
    p.add_argument('--lr', type=float, default=0.2)
    p.add_argument('--max-iter', type=int, default=2000)
    p.add_argument('--restarts', type=int, default=3)
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    if not args.preds_dir and not args.preds_csv:
        raise SystemExit('Either --preds-dir or --preds-csv must be provided')
    if args.preds_dir:
        probs, model_names, pred_game_ids = load_preds_from_dir(Path(args.preds_dir))
    else:
        probs, model_names, pred_game_ids = load_preds_from_csv(Path(args.preds_csv))
    y, y_game_ids = load_labels(Path(args.labels))
    aligned_probs, aligned_y = align_preds_labels(pred_game_ids, y_game_ids, probs, y)
    w, loss = optimize_weights(aligned_probs, aligned_y, lr=args.lr, max_iter=args.max_iter, restarts=args.restarts, verbose=args.verbose)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {model: float(wi) for model, wi in zip(model_names, w)}
    meta = {'loss': float(loss)}
    out.write_text(json.dumps({'weights': result, 'meta': meta}, indent=2))
    print('Saved weights to', out)


if __name__ == '__main__':
    main()
