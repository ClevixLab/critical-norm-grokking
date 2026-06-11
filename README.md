# The Weight Norm Sets the Grokking Timescale: A Causal Delay Law

Reproducibility repository for the paper
**"The Weight Norm Sets the Grokking Timescale: A Causal Delay Law."**

It contains every experiment runner, the **raw per-seed metrics** from every run reported in the paper,
and analysis scripts that regenerate **all headline numbers, tables, and figures** from those metrics.

One command reproduces the quantitative spine of the paper from the released data, on CPU, in seconds:

```bash
pip install -r requirements.txt        # numpy + matplotlib are enough for reproduction
python reproduce_all.py                # recompute numbers, print a report, regenerate figures
python reproduce_all.py --no-fig       # numbers only
```

Every recomputed value is printed next to the value stated in the paper, tagged `[OK ]` when it is within
3% and `[!! ]` otherwise, so any drift is visible at a glance.

## What the paper claims, and where each claim is reproduced

| Claim (paper) | Result | Data | Reproduced by |
|---|---|---|---|
| Concentrated critical norm under free dynamics, `‖W‖_c ∝ p^0.38` | Fig. 1, §4 | `data/grokfss/` | `analysis/make_figures.py` |
| Norm causally sets the grokking timescale (clamp matrix) | Fig. 2, Table 1, §5 | `data/efflr/` | `reproduce_all.py` → `rep_efflr` |
| Exponential delay law `T_grok ∝ e^{αρ}`, α≈7.64 over ρ∈[0.85,1.25] | Fig. 3, §5 | `data/denserho/` | `reproduce_all.py` → `rep_denserho` |
| Norm state vs. learning rate vs. transient dissociation | §5 controls | `data/normint/` | `reproduce_all.py` → `rep_normint` |
| Scaling law: shared exponent α≈7.49, R²=0.996 across p; single-exponent audit | Fig. 5, Table 3, §6 | `data/pscan/` | `reproduce_all.py` → `rep_pscan` |
| Second architecture (un-normalized transformer), α₂≈15.45 | Fig. 6, §7 | `data/tfnoln_long/` | `reproduce_all.py` → `rep_tfnoln_long` |
| Functional-norm threshold under LayerNorm | Fig. 7, §7 | `data/tfnormint/`, `data/tfgrok/` | `analysis/analyze_funcnorm.py` |
| Non-Fourier task (sparse parity) | Fig. 9, Table 2, §8 | `data/parity/`, `data/parity_lambda/` | `analysis/make_figures.py` |

See `MANIFEST.md` for the complete artifact → result map and `REPRODUCE.md` for step-by-step instructions
(including how to re-run experiments from scratch on a GPU).

## Repository layout

```
src/         experiment runners (one file per experiment; GPU, PyTorch)
analysis/    analysis + figure scripts that read data/<exp>/metrics/*.npz
data/        raw per-seed metrics (.npz) for every reported run + per-experiment grid_spec.json
figures/     the figures used in the paper (vector PDF + PNG previews)
paper/       final manuscript source (main.tex) and compiled paper.pdf
reproduce_all.py   one-command reproduction of the headline numbers and figures
```

## Two levels of reproduction

1. **From released metrics (default, recommended).** `python reproduce_all.py` reads the per-seed `.npz`
   files in `data/` and recomputes every fit, ratio, and exponent. No GPU, no training, ~seconds.
2. **From scratch.** Re-run any experiment with its runner in `src/` (PyTorch + CUDA). Each runner writes
   per-seed metrics into a `data/<experiment>/metrics/` folder in the same format the analysis scripts
   consume, so the level-1 pipeline then reproduces the paper from your fresh runs. See `REPRODUCE.md`.

## Data format

Each `data/<exp>/metrics/*.npz` holds per-seed arrays for one configuration — at minimum
`T_grok_per_seed` (grokking step per seed; non-positive = did not grok within budget), the recorded
`steps` grid, weight-norm trajectories, and the run's hyperparameters (`rho`, `lr`, `wd`, `p`, ...).
`data/<exp>/grid_spec.json` records the full configuration grid for that experiment.

## Citation

If you use this code or data, please cite the paper (see `CITATION.cff`).

## License

Code and data are released under the MIT License (`LICENSE`).
