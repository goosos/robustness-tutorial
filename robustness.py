"""
Parameter robustness — Goosos tutorial #9 companion module.

A parameter that works at 20 but breaks at 21 is not a finding,
it's noise. Good parameters sit on plateaus, not peaks.

Two tools:
  parameter_sensitivity() — slice the Sharpe grid along one axis and
      measure how fast performance degrades away from the optimum.
  find_plateau() — mark every grid cell within `frac` of the best
      Sharpe. A healthy strategy shows a broad plateau; a sharp
      lone peak is a red flag.

Works on any 2-D parameter grid: a pandas DataFrame with a
MultiIndex (param1, param2) and a 'sharpe' column, or anything
pivottable into that shape.
"""
import numpy as np
import pandas as pd


def as_grid(df, value_col="sharpe"):
    """Pivot a long-form grid into a 2-D DataFrame.

    Expects columns [param1, param2, value_col]; rows = param1,
    columns = param2.
    """
    p1 = df.columns[0]
    p2 = df.columns[1]
    return df.pivot_table(index=p1, columns=p2, values=value_col)


def parameter_sensitivity(grid, axis=0, at=None):
    """Sharpe profile along one parameter axis.

    grid: 2-D DataFrame (rows=param1, cols=param2) of Sharpe values.
    axis: 0 → vary param1 (fix param2 at `at`, default = argmax col);
          1 → vary param2 (fix param1 at `at`, default = argmax row).
    Returns a Series: index = parameter values, values = Sharpe.
    """
    if axis == 0:
        col = at if at is not None else grid.max().idxmax()
        return grid[col].dropna()
    col_idx = at if at is not None else grid.max(axis=1).idxmax()
    return grid.loc[col_idx].dropna()


def degradation_score(profile):
    """How fast does Sharpe fall away from the optimum?

    profile: Series from parameter_sensitivity().
    Returns the mean absolute Sharpe drop per unit step away from
    the best parameter, normalized by the best Sharpe. Lower =
    flatter (more robust). A cliff gives a high score.
    """
    vals = profile.values.astype(float)
    idx = profile.index.values.astype(float)
    best = int(np.nanargmax(vals))
    if vals[best] == 0 or not np.isfinite(vals[best]):
        return np.nan
    drops, steps = [], []
    for i in range(len(vals)):
        if i == best or not np.isfinite(vals[i]):
            continue
        drops.append((vals[best] - vals[i]) / abs(vals[best]))
        steps.append(abs(idx[i] - idx[best]))
    if not drops:
        return 0.0
    return float(np.mean(np.array(drops) / np.maximum(steps, 1e-9)))


def find_plateau(grid, frac=0.9):
    """Boolean mask of grid cells within `frac` of the best Sharpe.

    A broad connected plateau around the optimum = robust.
    A lone peak (optimum isolated, neighbors far below) = fragile.
    Returns (mask DataFrame, plateau_fraction, is_peak).
      plateau_fraction: share of grid cells on the plateau.
      is_peak: True if fewer than 5% of cells qualify — the
        optimum is a sharp lone spike.
    """
    best = grid.max().max()
    mask = grid >= frac * best
    frac_cells = float(mask.sum().sum() / mask.size)
    return mask, frac_cells, frac_cells < 0.05


def plateau_neighbors(grid, param, frac=0.9, radius=2, reference="global"):
    """Is `param` (a (p1, p2) tuple) sitting on a plateau?

    Checks the neighborhood within `radius` steps: what fraction
    of neighbors are within `frac` of the reference Sharpe?
    reference='global': compare to the grid's best Sharpe (is this
        param near the global optimum's plateau?).
    reference='local': compare to the param's own Sharpe (is this
        param on flat ground — robust to small tweaks?).
    Returns (neighbor_frac, verdict) where verdict is one of
    'plateau' (>=80% of neighbors qualify), 'slope' (40-80%),
    or 'peak' (<40% — fragile).
    """
    p1_vals = grid.index.values
    p2_vals = grid.columns.values
    i = int(np.argmin(np.abs(p1_vals - param[0])))
    j = int(np.argmin(np.abs(p2_vals - param[1])))
    sub = grid.iloc[max(0, i - radius):i + radius + 1,
                    max(0, j - radius):j + radius + 1]
    ref = grid.max().max() if reference == "global" else grid.iloc[i, j]
    if not np.isfinite(ref) or ref == 0:
        return 0.0, "peak"
    ok = float((sub >= frac * ref).sum().sum())
    frac_n = ok / sub.size
    verdict = "plateau" if frac_n >= 0.8 else ("slope" if frac_n >= 0.4 else "peak")
    return frac_n, verdict
