#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
analyze_scaling_variable.py  --  Is the grokking-delay exponent a single constant, and is rho=|W|/wc(p) the
right scaling variable? Pure reanalysis of the existing p-scan (no new training, no GPU).

It answers three questions a reviewer would press on:
  Q1  Is the per-p exponent alpha(p) really drifting, or is the drift within seed noise?
      -> bootstrap (resample seeds) a 95% CI on each per-p alpha and on the single shared alpha.
  Q2  Do per-p slopes (4 params) beat a single shared slope (1 param)?
      -> compare residual SSE; a tiny gain for 3 extra params means "shared" is adequate.
  Q3  Is rho=|W|/wc(p) the best collapse variable, or does another normalisation collapse better?
      -> compare global collapse R^2 across candidate scaling variables.

USAGE: python analyze_scaling_variable.py --root D:\Colab-local\paperA_delaylaw_<UTC>
       [--ps 41 53 59 71] [--rhos 0.85 1.00 1.15]
Reads the clamp + control cells already saved by run_delaylaw_pscan.py. Writes a JSON summary into --root.
"""
from __future__ import annotations
import argparse, glob, os, json
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ps", type=int, nargs="+", default=[41, 53, 59, 71])
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.85, 1.00, 1.15])
    ap.add_argument("--nboot", type=int, default=4000)
    a = ap.parse_args()
    M = os.path.join(a.root, "metrics"); rng = np.random.default_rng(0)
    ps, rlo = a.ps, a.rhos

    def seeds(p, rho):
        d = np.load(os.path.join(M, f"adamw_p{p}_a0.40_d128_H256_wd1e00_lr1e-03_clamp_rho{rho:.2f}.npz"),
                    allow_pickle=True)
        T = d["T_grok_per_seed"].astype(float); return T[T > 0]
    def wc(p):
        d = np.load(os.path.join(M, f"adamw_p{p}_a0.40_d128_H256_wd1e00_lr1e-03_ctrl.npz"), allow_pickle=True)
        w = d["wn_at_grok"]; return float(np.nanmedian(w[np.isfinite(w)]))

    raw = {(p, r): seeds(p, r) for p in ps for r in rlo}
    WC = {p: wc(p) for p in ps}
    med = {k: float(np.median(v)) for k, v in raw.items()}

    print("=== Q1: per-p exponent alpha(p) with bootstrap 95% CI (resample seeds) ===")
    rep = {"per_p_alpha": {}}
    for p in ps:
        a0 = np.polyfit(rlo, [np.log(med[(p, r)]) for r in rlo], 1)[0]
        bs = [np.polyfit(rlo, [np.log(np.median(rng.choice(raw[(p, r)], raw[(p, r)].size, replace=True)))
                               for r in rlo], 1)[0] for _ in range(a.nboot)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rep["per_p_alpha"][p] = [round(a0, 3), round(float(lo), 3), round(float(hi), 3)]
        print(f"  p={p}: alpha={a0:.2f}  95% CI [{lo:.2f}, {hi:.2f}]  (wc={WC[p]:.1f})")
    als = [rep["per_p_alpha"][p][0] for p in ps]
    cis = [rep["per_p_alpha"][p][1:] for p in ps]
    common_lo, common_hi = max(c[0] for c in cis), min(c[1] for c in cis)
    overlap = common_lo <= common_hi
    print(f"  CIs {'OVERLAP' if overlap else 'do NOT overlap'}"
          + (f' in [{common_lo:.2f}, {common_hi:.2f}] -> drift within noise, alpha consistent with constant'
             if overlap else ' -> drift is significant'))

    def shared_alpha(medd):
        rows = [(p, r, medd[(p, r)]) for p in ps for r in rlo]; pid = {p: i for i, p in enumerate(ps)}
        X = np.zeros((len(rows), 1 + len(ps))); y = np.zeros(len(rows))
        for i, (p, r, T) in enumerate(rows): X[i, 0] = r; X[i, 1 + pid[p]] = 1; y[i] = np.log(T)
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        R2 = 1 - np.sum((y - X @ c) ** 2) / np.sum((y - y.mean()) ** 2)
        return c[0], R2
    a_sh, R2_sh = shared_alpha(med)
    bs = [shared_alpha({k: np.median(rng.choice(raw[k], raw[k].size, replace=True)) for k in raw})[0]
          for _ in range(a.nboot)]
    print(f"\n=== shared single exponent: alpha={a_sh:.2f} [95% CI {np.percentile(bs,2.5):.2f}, "
          f"{np.percentile(bs,97.5):.2f}], collapse R^2={R2_sh:.4f} ===")
    rep["shared_alpha"] = [round(float(a_sh), 3), round(float(np.percentile(bs, 2.5)), 3),
                           round(float(np.percentile(bs, 97.5)), 3), round(float(R2_sh), 4)]

    print("\n=== Q2: per-p slopes (4 params) vs shared slope (1 param) ===")
    def sse(per_p):
        s = 0.0
        for p in ps:
            y = np.array([np.log(med[(p, r)]) for r in rlo]); x = np.array(rlo)
            if per_p: aa, bb = np.polyfit(x, y, 1)
            else: aa = a_sh; bb = np.mean(y - a_sh * x)
            s += float(np.sum((y - (aa * x + bb)) ** 2))
        return s
    s_sh, s_pp = sse(False), sse(True)
    rep["sse_shared"], rep["sse_per_p"] = round(s_sh, 4), round(s_pp, 4)
    print(f"  SSE shared={s_sh:.4f}  SSE per-p={s_pp:.4f}  -> per-p buys little for 3 extra params"
          f" ({'shared adequate' if s_pp > 0.5 * s_sh else 'per-p meaningfully better'})")

    print("\n=== Q3: best collapse variable (shared slope + per-p offset) ===")
    def collapse(xfun):
        rows = [(p, r, med[(p, r)]) for p in ps for r in rlo]; pid = {p: i for i, p in enumerate(ps)}
        X = np.zeros((len(rows), 1 + len(ps))); y = np.zeros(len(rows))
        for i, (p, r, T) in enumerate(rows): X[i, 0] = xfun(p, r); X[i, 1 + pid[p]] = 1; y[i] = np.log(T)
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        return 1 - np.sum((y - X @ c) ** 2) / np.sum((y - y.mean()) ** 2)
    cands = {"rho=|W|/wc(p) [ours]": lambda p, r: r,
             "|W| absolute":        lambda p, r: r * WC[p],
             "|W|/sqrt(p)":         lambda p, r: r * WC[p] / np.sqrt(p),
             "|W|/p^0.44":          lambda p, r: r * WC[p] / p ** 0.44}
    rep["collapse_R2"] = {}
    for lbl, xf in cands.items():
        R2 = float(collapse(xf)); rep["collapse_R2"][lbl] = round(R2, 4)
        print(f"  {lbl:22s}: collapse R^2={R2:.4f}")
    print("  -> if rho is within ~0.001 of the best, the relative-norm normalisation is justified.")
    json.dump(rep, open(os.path.join(a.root, "scaling_variable_audit.json"), "w"), indent=2)
    print(f"\nsaved -> {os.path.join(a.root, 'scaling_variable_audit.json')}")

if __name__ == "__main__":
    main()
