"""Charts for Part 9: plateau vs peak visualization."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

D = "/home/hatch/workspace/goosos-code/robustness-tutorial/"
df = pd.read_csv(D + "grid.csv")
grid = df.pivot_table(index="fast", columns="slow", values="sharpe")

# 1. Heatmap with MA(20,50) and grid-best marked
fig, ax = plt.subplots(figsize=(10, 7))
im = ax.imshow(grid.values, aspect="auto", origin="lower",
               extent=[grid.columns.min(), grid.columns.max(),
                       grid.index.min(), grid.index.max()],
               cmap="RdYlGn", vmin=-0.5, vmax=1.7)
fig.colorbar(im, ax=ax, label="Sharpe ratio")
ax.scatter([50], [20], s=120, c="white", edgecolors="black",
           linewidths=2, marker="o", label="MA(20,50) — plateau", zorder=5)
ax.scatter([33], [31], s=120, c="red", edgecolors="black",
           linewidths=2, marker="X", label="Grid best (31,33) — peak", zorder=5)
ax.set_xlabel("Slow window")
ax.set_ylabel("Fast window")
ax.set_title("MA parameter grid: plateau vs peak (SPY, 2023-10 → 2026-10)")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(D + "plateau_heatmap.png", dpi=100)
print("plateau_heatmap.png")


# 2. Sensitivity profiles: Sharpe vs fast (slow fixed) and vs slow (fast fixed)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

# Through MA(20,50)'s neighborhood: slow=50, vary fast
prof_f = grid[50].dropna()
axes[0].plot(prof_f.index, prof_f.values, "o-", ms=4, label="slow=50")
axes[0].axvline(20, color="black", ls="--", lw=1, label="fast=20")
axes[0].axhline(prof_f.loc[20] * 0.9, color="green", ls=":",
               lw=1, label="90% of MA(20,50)")
axes[0].fill_between(prof_f.index, prof_f.loc[20] * 0.9,
                     prof_f.values.max() * 1.05, color="green", alpha=0.1)
axes[0].set_xlabel("Fast window")
axes[0].set_ylabel("Sharpe ratio")
axes[0].set_title("Sensitivity: vary fast (slow=50)")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

# Through grid best: fast=31, vary slow
prof_s = grid.loc[31].dropna()
axes[1].plot(prof_s.index, prof_s.values, "o-", ms=4,
             color="red", label="fast=31")
axes[1].axvline(33, color="black", ls="--", lw=1, label="slow=33 (best)")
axes[1].axhline(prof_s.max() * 0.9, color="green", ls=":",
               lw=1, label="90% of best")
axes[1].set_xlabel("Slow window")
axes[1].set_title("Sensitivity: vary slow (fast=31) — the cliff")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

fig.suptitle("Plateau (left) vs cliff (right): same grid, different neighborhoods")
fig.tight_layout()
fig.savefig(D + "sensitivity_profiles.png", dpi=100)
print("sensitivity_profiles.png")
