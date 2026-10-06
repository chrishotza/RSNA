"""Calibrate local proxy changes against observed Kaggle public-LB changes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.ledger)
    required = ["proxy_silver", "public_lb"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"missing columns: {missing}")

    x = pd.to_numeric(df["proxy_silver"], errors="coerce")
    y = pd.to_numeric(df["public_lb"], errors="coerce")
    mask = np.isfinite(x) & np.isfinite(y)
    d = df.loc[mask].copy()

    print(f"usable submissions: {len(d)}")
    if len(d) < 3:
        print("Need at least 3 paired submissions for a first correlation check.")
        return

    x = pd.to_numeric(d["proxy_silver"], errors="coerce").to_numpy()
    y = pd.to_numeric(d["public_lb"], errors="coerce").to_numpy()

    print(f"Pearson(proxy, LB):  {pearsonr(x, y).statistic:.5f}")
    print(f"Spearman(proxy, LB): {spearmanr(x, y).statistic:.5f}")

    # Delta analysis is the primary calibration signal.
    if len(d) >= 4:
        xs = x[1:] - x[:-1]
        ys = y[1:] - y[:-1]
        print(f"Spearman(delta proxy, delta LB): {spearmanr(xs, ys).statistic:.5f}")
        print(f"Pearson(delta proxy, delta LB):  {pearsonr(xs, ys).statistic:.5f}")

    if len(d) >= 5:
        # Simple least-squares mapping. This is descriptive only until leave-one-out
        # validation is added; it must never be used to manufacture certainty.
        coef = np.polyfit(x, y, 1)
        print(f"Descriptive LB ~= {coef[0]:.5f} * proxy + {coef[1]:.5f}")


if __name__ == "__main__":
    main()
