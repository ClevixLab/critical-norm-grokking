# REPRODUCE — step by step

This repository reproduces *"The Weight Norm Sets the Grokking Timescale: A Causal Delay Law"* from the
released **raw per-seed metrics** in `data/`. There are two levels; most readers want Level 1.

## Level 1 — reproduce all numbers and figures from released metrics (CPU, seconds)

```bash
python -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt                       # numpy + matplotlib suffice for this level
python reproduce_all.py
```

Expected: a report in which every recomputed quantity is printed next to the paper value and tagged
`[OK ]` (within 3%). On the released data every line should read `[OK ]`. Figures
`figures/delay_dense_repro.pdf` and `figures/delay_transformer_repro.pdf` are regenerated from the raw
metrics as an independent visual check against the paper's `delay_dense.pdf` / `delay_transformer.pdf`.

To recompute numbers without touching figures:

```bash
python reproduce_all.py --no-fig
```

Individual results can also be regenerated with the per-experiment analysis scripts:

```bash
python analysis/analyze_denserho.py          # dense rho sweep: alpha, R^2
python analysis/analyze_delaylaw_pscan.py    # multi-p scaling collapse
python analysis/analyze_efflr_dissoc.py      # rho x lr clamp matrix (Table 1)
python analysis/analyze_tfnoln_long.py       # second-architecture delay law
python analysis/analyze_funcnorm.py          # LayerNorm functional-norm threshold
python analysis/reproduce_numbers.py         # consolidated numeric report
python analysis/make_figures.py              # regenerate the paper figures
```

## Level 2 — re-run experiments from scratch (GPU, PyTorch)

Each runner in `src/` regenerates the raw metrics for one experiment, writing
`data/<experiment>/metrics/*.npz` in the exact format the analysis scripts consume. After re-running, the
Level-1 pipeline reproduces the paper from your fresh runs.

```bash
pip install -r requirements.txt              # includes torch>=2.0 (needs a CUDA build for GPU)

python src/run_grokfss.py                    # observational critical norm  -> data/grokfss/
python src/run_efflr_dissoc.py               # causal clamp matrix (rho x lr) -> data/efflr/
python src/run_denserho.py                   # dense rho sweep              -> data/denserho/
python src/run_delaylaw_pscan.py             # multi-p scaling              -> data/pscan/
python src/run_normint.py                    # norm-state controls          -> data/normint/
python src/run_tfnoln_long.py                # un-normalized transformer    -> data/tfnoln_long/
python src/run_tfnormint.py                  # LayerNorm functional norm    -> data/tfnormint/
python src/run_parity.py                     # sparse parity                -> data/parity/
python src/run_parity_lambda.py              # parity weight-decay frontier -> data/parity_lambda/
```

Runners write atomically, support resume-after-interruption, and seed only from the base configuration
(excluding the intervention) so that a control and every intervention share initialization, data split, and
pre-intervention trajectory. The mapping from each runner to the result it produces is in `MANIFEST.md`.

## Environment

- Level 1: `numpy>=1.24`, `matplotlib>=3.7`. CPU only.
- Level 2: additionally `torch>=2.0` (CUDA build for GPU). Experiments in the paper were run on a single
  NVIDIA RTX-class GPU; budgets per experiment are recorded in each `data/<exp>/grid_spec.json`.

A fully reproducible environment is pinned in `requirements-lock.txt` (exact versions); `requirements.txt`
gives the minimal floor.
