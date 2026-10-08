"""
Part 9 demo: is MA(20,50) on a plateau or a peak?

Runs a focused MA(fast, slow) grid on SPY and applies the
robustness toolkit: sensitivity profiles, degradation scores,
plateau masks, and the neighborhood verdict for (20, 50).

Run:  python robustness_demo.py
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import vectorbt as vbt
import yfinance as yf

from robustness import (
    as_grid, parameter_sensitivity, degradation_score,
    find_plateau, plateau_neighbors,
)

SYMBOL = "SPY"
COMMISSION = 0.001


def load_data():
    df = yf.download(SYMBOL, period="3y", interval="1d",
                     auto_adjust=False, progress=False)
    return df["Close"][SYMBOL].dropna()


def run_grid(price):
    fast = np.arange(10, 41)    # 10..40
    slow = np.arange(30, 81)    # 30..80
    import itertools
    rows = []
    for fw, sw in itertools.product(fast, slow):
        if fw >= sw:
            continue
        f = price.rolling(fw).mean()
        s = price.rolling(sw).mean()
        entries = (f > s) & (f.shift(1) <= s.shift(1))
        exits = (f < s) & (f.shift(1) >= s.shift(1))
        pf = vbt.Portfolio.from_signals(
            price, entries.shift(1).fillna(False).astype(bool),
            exits.shift(1).fillna(False).astype(bool),
            fees=COMMISSION, freq="1D")
        rows.append((fw, sw, pf.sharpe_ratio()))
    return pd.DataFrame(rows, columns=["fast", "slow", "sharpe"])


def main():
    price = load_data()
    print(f"Data: {SYMBOL} daily, {len(price)} bars, "
          f"{price.index[0].date()} -> {price.index[-1].date()}")
    df = run_grid(price)
    print(f"Grid: {len(df)} combos (fast 10-40 x slow 30-80, fast<slow)")
    grid = as_grid(df)

    best = df.loc[df["sharpe"].idxmax()]
    print(f"\nBest in grid : fast={int(best['fast'])}, "
          f"slow={int(best['slow'])}, Sharpe={best['sharpe']:.2f}")

    # Sensitivity along each axis (through the best column/row)
    prof_fast = parameter_sensitivity(grid, axis=0)
    prof_slow = parameter_sensitivity(grid, axis=1)
    print(f"Fast-axis degradation: {degradation_score(prof_fast):.4f} "
          f"(Sharpe loss per step from optimum)")
    print(f"Slow-axis degradation: {degradation_score(prof_slow):.4f}")

    # Plateau analysis
    mask, frac_cells, is_peak = find_plateau(grid, frac=0.9)
    print(f"\nPlateau (>=90% of best): {frac_cells:.1%} of grid cells, "
          f"is_peak={is_peak}")

    for label, params in [("MA(20,50)", (20, 50)),
                          ("grid best", (int(best['fast']), int(best['slow'])))]:
        nf_g, verdict_g = plateau_neighbors(grid, params, frac=0.9, radius=2,
                                            reference="global")
        nf_l, verdict_l = plateau_neighbors(grid, params, frac=0.9, radius=2,
                                            reference="local")
        own = grid.loc[params[0], params[1]] if params[0] in grid.index and params[1] in grid.columns else float("nan")
        print(f"{label} (Sharpe={own:.2f}): "
              f"global {nf_g:.0%}->{verdict_g.upper()}, "
              f"local {nf_l:.0%}->{verdict_l.upper()}")

    # Save grid for charts
    df.to_csv("/home/hatch/workspace/goosos-code/robustness-tutorial/"
              "grid.csv", index=False)
    print("\nSaved grid.csv")


if __name__ == "__main__":
    main()
