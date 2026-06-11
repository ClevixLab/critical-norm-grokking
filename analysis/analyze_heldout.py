#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
analyze_heldout.py  --  Score the PREREGISTERED held-out prediction at p=67 (interp.) and p=97 (extrap.).

USAGE:
  python analyze_heldout.py --root D:\Colab-local\paperA_heldout_<UTC> --pred heldout_prediction_<UTC>.json

Primary (prefactor-independent) test:
  Because the prediction is reparametrised around rho=1, the two models share L1(p) and differ only in the
  SLOPE alpha. The clean, L1-independent discriminator is therefore the observed exponent itself:
      alpha_obs(p*) = [log T(1.15) - log T(0.85)] / 0.30,
  compared to alpha_drift (M1) and alpha_universal=7.49 (M2), with a bootstrap CI on alpha_obs.
Secondary test:
  Does the observed curve fall inside each model's preregistered 95% band (this also folds in the L1(p)
  level-extrapolation, which is shared by both models)?
Also: measured ||W||_c(p*) vs the preregistered p^k prediction.
"""
from __future__ import annotations
import argparse, glob, os, json
import numpy as np

RLO = [0.85, 1.00, 1.15]

def seed_times(M, p, rho):
    f = os.path.join(M, f"adamw_p{p}_a0.40_d128_H256_wd1e00_lr1e-03_clamp_rho{rho:.2f}.npz")
    d = np.load(f, allow_pickle=True); T = d["T_grok_per_seed"].astype(float)
    return T[T > 0], int(d["max_steps"]), int((T > 0).sum()), len(T), float(np.nanmean(d["wn_at_grok"]))

def wc_of(M, p):
    d = np.load(os.path.join(M, f"adamw_p{p}_a0.40_d128_H256_wd1e00_lr1e-03_ctrl.npz"), allow_pickle=True)
    w = d["wn_at_grok"]; return float(np.nanmedian(w[np.isfinite(w)]))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--pred", required=True)
    a = ap.parse_args(); M = os.path.join(a.root, "metrics"); rng = np.random.default_rng(7)
    pred = json.load(open(a.pred))
    print(f"Scoring against preregistration locked {pred['created_utc']}")
    print(f"(trained on p={pred['trained_on_p']}, models: {pred['meta']['L1_model'][0]}/{pred['meta']['alpha_model'][0]})\n")
    for ps in pred["targets"]:
        ps = int(ps)
        try:
            cells = {r: seed_times(M, ps, r) for r in RLO}
        except FileNotFoundError:
            print(f"p={ps}: data not present yet."); continue
        med = {r: float(np.median(cells[r][0])) for r in RLO}
        tag = "interpolation" if ps < 71 else "EXTRAPOLATION"
        print(f"=== p={ps} ({tag}) ===")
        for r in RLO:
            tg, bud, g, S, _ = cells[r]
            cen = "" if tg.max() < 0.95 * bud else "  <-CENSORED"
            print(f"  rho={r:.2f}: T_obs={med[r]:7.0f}  grok {g}/{S}  max {int(tg.max())}/{bud}{cen}")
        # observed alpha (L1-independent slope) + bootstrap CI
        a_obs = (np.log(med[1.15]) - np.log(med[0.85])) / 0.30
        boots = []
        for _ in range(3000):
            m085 = np.median(rng.choice(cells[0.85][0], cells[0.85][0].size, replace=True))
            m115 = np.median(rng.choice(cells[1.15][0], cells[1.15][0].size, replace=True))
            boots.append((np.log(m115) - np.log(m085)) / 0.30)
        alo, ahi = np.percentile(boots, [2.5, 97.5])
        pr = pred["predictions"][str(ps)]
        ad, au = pr["alpha_drift"], pr["alpha_universal"]
        print(f"  alpha_obs = {a_obs:.2f}  [95% CI {alo:.2f}, {ahi:.2f}]")
        print(f"     vs  M1 drift alpha={ad:.2f}   |   M2 universal alpha={au:.2f}")
        favor = "M2 (universal/shared exponent)" if abs(a_obs - au) < abs(a_obs - ad) else "M1 (alpha drifts with p)"
        in_d = alo <= ad <= ahi; in_u = alo <= au <= ahi
        verdict = favor
        if in_d and in_u: verdict += " — but CI contains both; inconclusive discrimination"
        elif in_u and not in_d: verdict = "M2 universal (CI excludes drift)"
        elif in_d and not in_u: verdict = "M1 drift (CI excludes universal)"
        print(f"  --> favors {verdict}")
        # secondary: band membership (level + slope)
        for mdl in ("M1_drift", "M2_universal"):
            inside = all(pr[mdl + "_CI"][f"{r:.2f}"][0] <= med[r] <= pr[mdl + "_CI"][f"{r:.2f}"][1] for r in RLO)
            looser = all(0.7 * pr[mdl][f"{r:.2f}"] <= med[r] <= 1.4 * pr[mdl][f"{r:.2f}"] for r in RLO)
            print(f"  curve within {mdl} 95% band: {inside}   (within +/-30-40%: {looser})")
        # wc check
        wcm = wc_of(M, ps); wcp = pred["wc_scaling"]["predicted_wc"][str(ps)]
        print(f"  ||W||_c measured={wcm:.1f}  vs preregistered p^k prediction={wcp:.1f}  (ratio {wcm/wcp:.2f})\n")

if __name__ == "__main__":
    main()
