"""
generate_report.py — Evaluate trained MCP Cost Estimator models and produce
a structured accuracy report.

Usage:
    python generate_report.py

Output:
    • Console: formatted table with per-combo metrics
    • reports/model_accuracy_report.csv  — machine-readable version
    • reports/model_accuracy_report.md   — markdown table
"""

import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ---------------------------------------------------------------------------
# Import data-loading and encoding constants from train_model.py
# ---------------------------------------------------------------------------

from train_model import (
    COMBO_DATA,
    FEATURE_NAMES,
    MODELS_DIR,
    load_combo_data,
)

REPORTS_DIR = Path(__file__).parent / "reports"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, R², MAPE in raw token space."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
    # MAPE — guard against zero actuals
    nonzero = y_true != 0
    mape = (
        np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
        if nonzero.any() else float("nan")
    )
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def cv_mae(X: pd.DataFrame, y_log: np.ndarray, y_raw: np.ndarray,
           pipeline, n_splits: int) -> float:
    """Return mean CV-MAE in raw token space using k-fold CV."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_maes = []
    for train_idx, test_idx in kf.split(X):
        pipeline.fit(X.iloc[train_idx], y_log[train_idx])
        y_pred = np.expm1(pipeline.predict(X.iloc[test_idx])).clip(0)
        fold_maes.append(mean_absolute_error(y_raw[test_idx], y_pred))
    # Refit on full data to restore original model state
    pipeline.fit(X, y_log)
    return float(np.mean(fold_maes))


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def build_report() -> list[dict]:
    rows = []

    for (server, dataset) in COMBO_DATA:
        records = load_combo_data(server, dataset)
        if len(records) < 3:
            print(f"  SKIP  ({server}, {dataset}) — only {len(records)} samples")
            continue

        df = pd.DataFrame(records)
        X  = df[FEATURE_NAMES]
        n  = len(records)
        n_splits = min(5, max(2, n // 3))

        for target in ("input_tokens", "output_tokens"):
            model_path = MODELS_DIR / f"{server}_{dataset}_{target}_model.joblib"
            if not model_path.exists():
                print(f"  MISSING model: {model_path.name}")
                continue

            pipeline = joblib.load(model_path)
            y_raw = df[target].values.astype(float)
            y_log = np.log1p(y_raw)

            # Train-set metrics
            y_pred_log  = pipeline.predict(X)
            y_pred_raw  = np.expm1(y_pred_log).clip(0)
            train_stats = compute_metrics(y_raw, y_pred_raw)

            # CV MAE (out-of-sample estimate)
            cv_mae_val = cv_mae(X, y_log, y_raw, pipeline, n_splits) if n >= n_splits * 2 else float("nan")

            rows.append({
                "server":       server,
                "dataset":      dataset,
                "target":       target,
                "n_samples":    n,
                "train_mae":    round(train_stats["mae"],  1),
                "train_rmse":   round(train_stats["rmse"], 1),
                "train_r2":     round(train_stats["r2"],   3),
                "train_mape":   round(train_stats["mape"], 1),
                "cv_mae":       round(cv_mae_val, 1) if not np.isnan(cv_mae_val) else "—",
            })
            print(
                f"  OK    ({server:12s}, {dataset:7s}) {target:14s}  "
                f"n={n:2d}  train_MAE={train_stats['mae']:7.0f}  "
                f"cv_MAE={cv_mae_val:7.0f}  R²={train_stats['r2']:+.3f}"
            )

    return rows


def format_table(rows: list[dict]) -> str:
    """Return a human-readable fixed-width table."""
    header = (
        f"{'Server':<12}  {'Dataset':<7}  {'Target':<14}  "
        f"{'N':>2}  {'Train MAE':>9}  {'Train RMSE':>10}  "
        f"{'Train R²':>8}  {'Train MAPE%':>11}  {'CV MAE':>7}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    prev_combo = None
    for r in rows:
        combo = (r["server"], r["dataset"])
        if prev_combo and combo != prev_combo:
            lines.append("")
        lines.append(
            f"{r['server']:<12}  {r['dataset']:<7}  {r['target']:<14}  "
            f"{r['n_samples']:>2}  {r['train_mae']:>9.0f}  {r['train_rmse']:>10.0f}  "
            f"{r['train_r2']:>+8.3f}  {str(r['train_mape']):>11}  {str(r['cv_mae']):>7}"
        )
        prev_combo = combo
    lines.append(sep)
    return "\n".join(lines)


def save_csv(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nCSV  → {path}")


def save_markdown(rows: list[dict], path: Path) -> None:
    df = pd.DataFrame(rows)
    df.columns = [c.replace("_", " ").title() for c in df.columns]
    md = df.to_markdown(index=False)
    path.write_text(md, encoding="utf-8")
    print(f"MD   → {path}")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("\nEvaluating models...\n")
    rows = build_report()

    if not rows:
        print("\nNo models evaluated — run train_model.py first.")
        return

    print("\n" + "=" * 90)
    print("  MODEL ACCURACY REPORT")
    print("=" * 90 + "\n")
    print(format_table(rows))

    # Aggregate summary
    df = pd.DataFrame(rows)
    print("\nSUMMARY (mean across all combos)")
    print(f"  Input  tokens — Train MAE: {df[df.target=='input_tokens']['train_mae'].mean():,.0f}  "
          f"CV MAE: {pd.to_numeric(df[df.target=='input_tokens']['cv_mae'], errors='coerce').mean():,.0f}  "
          f"R²: {df[df.target=='input_tokens']['train_r2'].mean():+.3f}")
    print(f"  Output tokens — Train MAE: {df[df.target=='output_tokens']['train_mae'].mean():,.0f}  "
          f"CV MAE: {pd.to_numeric(df[df.target=='output_tokens']['cv_mae'], errors='coerce').mean():,.0f}  "
          f"R²: {df[df.target=='output_tokens']['train_r2'].mean():+.3f}")

    save_csv(rows, REPORTS_DIR / "model_accuracy_report.csv")
    save_markdown(rows, REPORTS_DIR / "model_accuracy_report.md")


if __name__ == "__main__":
    main()
