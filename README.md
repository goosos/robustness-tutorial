# Parameter Robustness Tutorial

Part 9 of [Build Your Own Quant Research System](https://goosos.com/tutorials/).
Article: https://goosos.com/parameter-robustness (draft)

A parameter that works at 20 but breaks at 21 is noise, not a finding.
Good parameters sit on plateaus, not peaks.

## What this does

`robustness_demo.py` runs 1,515 MA(fast, slow) combinations on SPY
(2023-10 → 2026-10) and applies the robustness toolkit:

| Parameter | Sharpe | Local verdict |
|---|---|---|
| Grid best (31, 33) | 1.62 | **PEAK** (12% of neighbors) |
| MA(20, 50) | 0.92 | **PLATEAU** (100% of neighbors) |

The "best" parameters are a fragile spike — fast=31, slow=33 are
suspiciously close together (whipsaw luck). MA(20,50) is nowhere near
optimal, but every nearby parameter works about as well. Boring beats
brilliant.

## Files

- `robustness.py` — `parameter_sensitivity()`, `degradation_score()`,
  `find_plateau()`, `plateau_neighbors()` (global + local reference modes)
- `robustness_demo.py` — full demo (downloads SPY, runs grid, prints verdicts)
- `make_charts.py` — heatmap + sensitivity profile charts
- `article.md` — the tutorial text

## Run it

```bash
pip install -r requirements.txt
python robustness_demo.py
```

## The lesson

Two questions, two reference modes: "am I near the optimum?" (global)
vs "am I on flat ground?" (local). For real money, the second question
is the one that matters. Never deploy a `"peak"` verdict, no matter how
good the headline Sharpe looks.

## Toolkit

This is a synthesis part — no new module. `robustness.py` composes
existing [quant-toolkit](https://github.com/goosos/quant-toolkit) modules
(`backtest`, `metrics`, `overfitting`). The plateau test is the practical
sibling of Part 4's PBO: PBO asks "did I overfit by *selecting*?", the
plateau test asks "did I overfit by *locating*?"
