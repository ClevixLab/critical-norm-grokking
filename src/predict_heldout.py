#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
predict_heldout.py  --  PREREGISTERED held-out prediction of the grokking delay at unseen task sizes.

Run this on the EXISTING 4-p delay-law scan (p in {41,53,59,71}) BEFORE collecting p=67/97 data. It locks
predictions to a timestamped JSON so the held-out test is genuinely out-of-sample (no post-hoc tuning).

Method (reparametrised around rho=1, the centre of the rho grid, to minimise rho-extrapolation error):
  log T_grok(p, rho) = alpha(p) * (rho - 1) + L1(p),   L1(p) := log T_grok(p, rho=1).
We model the two scalar functions on the 4 known p and predict at p* in {67 (interp.), 97 (extrap.)}:
  * L1(p): grok time at rho=1 (shared across both models). Fit log T1 vs p (linear) and vs log p; keep better R^2.
  * alpha(p): TWO competing models, which the held-out point will adjudicate ---
        M1 "drift":     alpha(p) linear in p, extrapolated (assumes the observed mild decrease continues);
        M2 "universal": alpha = 7.49 constant (the shared-exponent / scaling-law hypothesis).
  At p=67 (inside the fitted range) M1 and M2 nearly agree -> consistency check.
  At p=97 (outside) they diverge -> the data discriminate drift vs universality.
95% bands: resample the per-cell seed-level T_grok (nonparametric bootstrap), refit everything, re-predict.

USAGE: python predict_heldout.py --root D:\Colab-local\paperA_delaylaw_<UTC>  [--out predictions_dir]
Output: <out>/heldout_prediction_<UTC>.json  (the locked preregistration).
"""
from __future__ import annotations
import argparse, glob, os, json, datetime
import numpy as np

PS = [41, 53, 59, 71]
RLO = [0.85, 1.00, 1.15]
TARGETS = [67, 97]
ALPHA_UNIVERSAL = 7.49

def seed_times(M, p, rho):
    d = np.load(os.path.join(M, f"adamw_p{p}_a0.40_d128_H256_wd1e00_lr1e-03_clamp_rho{rho:.2f}.npz"),
                allow_pickle=True)
    T = d["T_grok_per_seed"].astype(float); return T[T > 0]

def wc_of(M, p):
    d = np.load(os.path.join(M, f"adamw_p{p}_a0.40_d128_H256_wd1e00_lr1e-03_ctrl.npz"), allow_pickle=True)
    w = d["wn_at_grok"]; return float(np.nanmedian(w[np.isfinite(w)]))

def better_linfit(x, y):
    """Return a predictor f(x*) using whichever of {linear in x, linear in log x} fits the 4 points better."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    def fit(xx):
        a, b = np.polyfit(xx, y, 1); pred = a * xx + b
        R2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        return (a, b, R2)
    lin = fit(x); log = fit(np.log(x))
    if log[2] > lin[2]:
        return (lambda xs: log[0] * np.log(xs) + log[1]), ("logp", log[2])
    return (lambda xs: lin[0] * xs + lin[1]), ("linp", lin[2])

def fit_predict(med):
    """med[p] = [T(0.85), T(1.0), T(1.15)] medians. Returns per-model predictions at TARGETS."""
    alpha = {}; L1 = {}
    for p in PS:
        a, b = np.polyfit(RLO, np.log(med[p]), 1); alpha[p] = a; L1[p] = a * 1.0 + b
    L1_pred, L1_how = better_linfit(PS, [L1[p] for p in PS])
    al_pred, al_how = better_linfit(PS, [alpha[p] for p in PS])
    out = {}
    for ps in TARGETS:
        L1s = float(L1_pred(ps))
        a_drift = float(al_pred(ps))
        out[ps] = {
            "L1": L1s, "alpha_drift": a_drift, "alpha_universal": ALPHA_UNIVERSAL,
            "M1_drift":     {f"{r:.2f}": float(np.exp(a_drift * (r - 1) + L1s)) for r in RLO},
            "M2_universal": {f"{r:.2f}": float(np.exp(ALPHA_UNIVERSAL * (r - 1) + L1s)) for r in RLO},
        }
    return out, dict(L1_model=L1_how, alpha_model=al_how, per_p_alpha={p: round(alpha[p], 3) for p in PS})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--nboot", type=int, default=3000)
    a = ap.parse_args()
    M = os.path.join(a.root, "metrics"); rng = np.random.default_rng(20260610)
    raw = {p: {r: seed_times(M, p, r) for r in RLO} for p in PS}
    med = {p: [float(np.median(raw[p][r])) for r in RLO] for p in PS}
    point, meta = fit_predict(med)

    # bootstrap bands
    boot = {ps: {"M1_drift": {f"{r:.2f}": [] for r in RLO},
                 "M2_universal": {f"{r:.2f}": [] for r in RLO}} for ps in TARGETS}
    for _ in range(a.nboot):
        bmed = {p: [float(np.median(rng.choice(raw[p][r], raw[p][r].size, replace=True))) for r in RLO] for p in PS}
        bp, _ = fit_predict(bmed)
        for ps in TARGETS:
            for r in RLO:
                k = f"{r:.2f}"
                boot[ps]["M1_drift"][k].append(bp[ps]["M1_drift"][k])
                boot[ps]["M2_universal"][k].append(bp[ps]["M2_universal"][k])
    def band(v): return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    for ps in TARGETS:
        for mdl in ("M1_drift", "M2_universal"):
            for r in RLO:
                k = f"{r:.2f}"; point[ps][mdl + "_CI"] = point[ps].get(mdl + "_CI", {})
                point[ps][mdl + "_CI"][k] = band(boot[ps][mdl][k])

    # wc scaling (secondary check), report exponent + predictions, but we will MEASURE wc from control too
    wcs = [wc_of(M, p) for p in PS]; k, c = np.polyfit(np.log(PS), np.log(wcs), 1)
    wc_pred = {ps: float(np.exp(k * np.log(ps) + c)) for ps in TARGETS}

    rec = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "note": "PREREGISTERED. Predictions locked before p=67/97 data collected.",
        "source_root": os.path.basename(a.root.rstrip("/\\")),
        "trained_on_p": PS, "rho_grid": RLO, "targets": TARGETS,
        "meta": meta, "measured_med_T1_per_p": {p: med[p][1] for p in PS},
        "predictions": point,
        "wc_scaling": {"exponent_k": float(k), "predicted_wc": wc_pred,
                       "note": "secondary; wc will also be measured from each target's control"},
        "adjudication_rule": ("At p=67 (interpolation) M1 and M2 should both contain the observed curve. "
                              "At p=97 (extrapolation) they diverge: if observed alpha ~7.5 and the curve falls "
                              "in M2's band -> supports a shared (universal) exponent; if it falls in M1's band "
                              "(lower alpha, faster grok) -> the alpha-drift with p is real. Report whichever, honestly."),
    }
    outdir = a.out or a.root
    os.makedirs(outdir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(outdir, f"heldout_prediction_{stamp}.json")
    json.dump(rec, open(path, "w"), indent=2)
    # human-readable echo
    print(f"LOCKED preregistration -> {path}")
    print(f"models: L1 {meta['L1_model']}, alpha {meta['alpha_model']}; per-p alpha {meta['per_p_alpha']}")
    for ps in TARGETS:
        tag = "interp." if ps < 71 else "EXTRAP."
        pr = point[ps]
        print(f"\n p={ps} ({tag}):  alpha_drift={pr['alpha_drift']:.2f}  alpha_univ={ALPHA_UNIVERSAL}")
        for r in RLO:
            k2 = f"{r:.2f}"
            print(f"   rho={k2}:  M1_drift={pr['M1_drift'][k2]:7.0f} {pr['M1_drift_CI'][k2]}"
                  f"   M2_univ={pr['M2_universal'][k2]:7.0f} {pr['M2_universal_CI'][k2]}")
        print(f"   wc_pred={wc_pred[ps]:.1f} (p^{k:.3f}; will be measured too)")

if __name__ == "__main__":
    main()
