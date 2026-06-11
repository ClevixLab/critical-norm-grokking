# MANIFEST — what reproduces what

Every headline number, table, and figure in the paper traces to a `data/<experiment>/metrics/` folder of
**per-seed** `.npz` files plus a script. `python reproduce_all.py` recomputes the core spine (Figs 2, 3, 5, 6
and Tables 1, 3) directly from the raw metrics and prints each value next to the one stated in the paper.

## Result → data → runner → analysis

| Paper result | `data/` folder | Runner (`src/`) | Analysis |
|---|---|---|---|
| Fig. 1, §4 — critical norm under free dynamics; `‖W‖_c ∝ p^0.38` | `grokfss/` | `run_grokfss.py` | `make_figures.py` |
| Fig. 2, Table 1, §5 — causal clamp matrix (ρ × lr, p=59) | `efflr/` | `run_efflr_dissoc.py` | `analyze_efflr_dissoc.py`; `reproduce_all.py:rep_efflr` |
| Fig. 3, §5 — dense ρ sweep, α≈7.64, R²=0.994 | `denserho/` | `run_denserho.py` | `analyze_denserho.py`; `reproduce_all.py:rep_denserho` |
| §5 controls — total norm vs. inter-layer vs. transient; one-shot washout | `normint/` | `run_normint.py`, `run_normint_control.py` | `reproduce_all.py:rep_normint` |
| Fig. 5, Table 3, §6 — scaling collapse, shared α≈7.49, single-exponent audit | `pscan/` | `run_delaylaw_pscan.py` | `analyze_delaylaw_pscan.py`, `analyze_scaling_variable.py`; `reproduce_all.py:rep_pscan` |
| Fig. 6, §7 — second architecture (un-normalized transformer), α₂≈15.45 | `tfnoln_long/` (and `tfnoln/`) | `run_tfnoln_long.py`, `run_tfnoln.py` | `analyze_tfnoln_long.py`; `reproduce_all.py:rep_tfnoln_long` |
| Fig. 7, §7 — LayerNorm functional-norm threshold (unembedding) | `tfnormint/`, `tfgrok/` | `run_tfnormint.py`, `run_tfgrok.py` | `analyze_funcnorm.py` |
| Fig. 9, Table 2, §8 — sparse parity (non-Fourier); λ frontier | `parity/`, `parity_lambda/` | `run_parity.py`, `run_parity_lambda.py` | `make_figures.py` |
| §4 critical-norm finite-size sweep (held-out checks) | `critfss/` | `run_critfss.py` | `predict_heldout.py`, `analyze_heldout.py`, `run_heldout.py` |

## `.npz` schema

Each metrics file is one configuration. Keys present across experiments include:

- `T_grok_per_seed` — grokking step per seed (≤0 means did not grok within budget)
- `steps` — the step grid on which trajectories are logged
- `wn_at_grok` — weight norm at the grokking step (per seed)
- `wn_E`, `wn_W1`, `wn_W2` — per-layer norm trajectories (where applicable)
- hyperparameters as scalars: `rho`, `lr`, `wd`, `p`, `arm`, `wc_used`, ...

`data/<exp>/grid_spec.json` records the full configuration grid and budget for that experiment;
`data/<exp>/results_summary.json` holds the per-experiment summary the runner emitted.

## Determinism

Analysis bootstraps use a fixed seed (`numpy.random.default_rng(0)` in `reproduce_all.py`), so the
confidence intervals printed by the reproduction are themselves reproducible bit-for-bit.
