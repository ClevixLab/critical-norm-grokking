#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
analyze_funcnorm.py  --  Offline read-outs for run_funcnorm.py outputs.

USAGE (PowerShell):
  python analyze_funcnorm.py --root D:\Colab-local\paperA_funcnorm_2026...

Produces (printed + saved to <root>\funcnorm_analysis.json):

(a) SCALE  -> the function-scale critical-norm law
    For each candidate norm at grokking, across primes p:
      * mean value, CV across seeds (concentration)
      * power-law fit  norm_c(p) ~ A * p^beta   (R^2, beta +/- se)
    Candidates: total ||W||, ||E||, ||W1||, ||W2|| (extensive)
                e_rowrms=||E||/sqrt(p), w2_colrms=||W2||/sqrt(p), e_elemrms (intensive)
    KEY QUESTIONS the table answers:
      - which norm is the *tightest* invariant (lowest CV)?  [expect embedding-family]
      - which intensive norm is *scale-free* (beta ~ 0, p-invariant)?  [the fundamental scale]

(b) OPTWD -> is the critical norm specific to AdamW + weight decay?
    For each (optimiser, weight-decay, lr) condition:
      * grok fraction (how many seeds grokked within budget)
      * ||W||@grok and ||E||@grok (and CV) -- is the function-scale norm the SAME as AdamW+wd?
      * norm DIRECTION: ||.||@mem vs ||.||@grok  -> did the network reach grokking from BELOW
        (norm rose, Golechha/Notsawo "increasing-norm" regime) or from ABOVE (norm fell,
        Omnigrok relaxation)?  An attractor approached from either direction is the strong claim.
"""
from __future__ import annotations
import argparse, json, glob, os
import numpy as np

def load(f): return np.load(f, allow_pickle=True)

def fit_powerlaw(p, y):
    p = np.asarray(p, float); y = np.asarray(y, float)
    m = (y > 0) & np.isfinite(y)
    if m.sum() < 3: return dict(beta=np.nan, R2=np.nan, A=np.nan, n=int(m.sum()))
    x = np.log(p[m]); ly = np.log(y[m])
    A = np.vstack([x, np.ones_like(x)]).T
    (beta, b), *_ = np.linalg.lstsq(A, ly, rcond=None)
    pred = A @ [beta, b]; R2 = 1 - np.sum((ly - pred) ** 2) / np.sum((ly - ly.mean()) ** 2)
    return dict(beta=round(float(beta), 3), R2=round(float(R2), 4), A=round(float(np.exp(b)), 3), n=int(m.sum()))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); a = ap.parse_args()
    files = glob.glob(os.path.join(a.root, "metrics", "*.npz"))
    scale, optwd = [], []
    for f in files:
        d = load(f)
        role = str(d['role'])
        T = d['T_grok_per_seed']; g = (T > 0)
        rec = dict(p=int(d['p']), optim=str(d['optim']), lam=float(d['lam']), lr=float(d['lr']),
                   grok=int(g.sum()), S=int(len(T)))
        for key in ['weight_norm', 'wn_E', 'wn_W1', 'wn_W2', 'e_rowrms', 'w2_colrms', 'e_elemrms']:
            ag = d[f'{key}__at_grok'] if f'{key}__at_grok' in d.files else None
            if ag is None:  # e_elemrms has no __at_grok stored; reconstruct from wn_E
                continue
            v = ag[np.isfinite(ag)]
            rec[f'{key}_mean'] = float(np.mean(v)) if len(v) else np.nan
            rec[f'{key}_cv'] = float(100 * np.std(v) / np.mean(v)) if len(v) else np.nan
        # norm direction (mem -> grok)
        for key in ['weight_norm', 'wn_E']:
            am = d.get(f'{key}__at_mem'); ag = d.get(f'{key}__at_grok')
            if f'{key}__at_mem' in d.files and f'{key}__at_grok' in d.files:
                am = d[f'{key}__at_mem']; ag = d[f'{key}__at_grok']
                mask = np.isfinite(am) & np.isfinite(ag)
                rec[f'{key}_mem'] = float(np.mean(am[mask])) if mask.any() else np.nan
                rec[f'{key}_grok'] = float(np.mean(ag[mask])) if mask.any() else np.nan
        (scale if role == 'scale' else optwd).append(rec)

    out = {}
    # ---- (a) scaling table + fits ----
    if scale:
        scale = sorted(scale, key=lambda r: r['p'])
        ps = [r['p'] for r in scale]
        print("\n=== (a) SCALE: norm@grok across p  (mean | CV%) ===")
        cand = [('weight_norm', 'total||W||'), ('wn_E', '||E||'), ('wn_W2', '||W2||'),
                ('e_rowrms', '||E||/sqrt(p)'), ('w2_colrms', '||W2||/sqrt(p)')]
        hdr = "  p   grok  " + "".join(f"{lab:>16s}" for _, lab in cand)
        print(hdr)
        for r in scale:
            line = f"  {r['p']:3d}  {r['grok']:2d}/{r['S']:<2d} "
            for k, _ in cand:
                line += f"  {r.get(k+'_mean', float('nan')):7.2f}({r.get(k+'_cv', float('nan')):4.1f}%)"
            print(line)
        print("\n  power-law fits  norm_c(p) ~ A p^beta  (beta -> 0 means scale-free / p-invariant):")
        fits = {}
        for k, lab in cand:
            fit = fit_powerlaw(ps, [r.get(k + '_mean', np.nan) for r in scale])
            fits[k] = fit
            print(f"    {lab:16s}: beta={fit['beta']:+.3f}  R2={fit['R2']:.4f}  (A={fit['A']})")
        # tightest invariant = min mean CV across p
        meancv = {k: float(np.nanmean([r.get(k + '_cv', np.nan) for r in scale])) for k, _ in cand}
        tightest = min(meancv, key=meancv.get)
        print(f"\n  tightest invariant (lowest mean CV across p): {tightest}  (mean CV {meancv[tightest]:.2f}%)")
        out['scale'] = dict(table=scale, fits=fits, mean_cv=meancv, tightest=tightest)

    # ---- (b) optimiser x weight-decay ----
    if optwd:
        print("\n=== (b) OPTWD: does it grok, and where? ===")
        print("  optim  wd    lr     grok    ||W||:mem->grok    ||E||:mem->grok   (direction)")
        for r in sorted(optwd, key=lambda r: (r['optim'], -r['lam'], r['lr'])):
            wm, wg = r.get('weight_norm_mem', np.nan), r.get('weight_norm_grok', np.nan)
            em, eg = r.get('wn_E_mem', np.nan), r.get('wn_E_grok', np.nan)
            arrow = '—'
            if np.isfinite(wm) and np.isfinite(wg):
                arrow = 'ROSE' if wg > wm * 1.02 else ('FELL' if wg < wm * 0.98 else 'flat')
            print(f"  {r['optim']:5s}  {r['lam']:.1f}  {r['lr']:.0e}  {r['grok']:2d}/{r['S']:<2d}  "
                  f"{wm:6.2f}->{wg:6.2f}   {em:6.2f}->{eg:6.2f}   {arrow}")
        print("\n  Interpretation: if a no-WD or SGD condition groks and ||E||@grok matches the "
              "AdamW+wd value\n  while the norm ROSE to it, the function-scale critical norm is an "
              "attractor approached\n  from EITHER direction (reconciles Omnigrok vs Golechha/Notsawo).")
        out['optwd'] = optwd

    json.dump(out, open(os.path.join(a.root, "funcnorm_analysis.json"), "w"), indent=1)
    print(f"\nsaved -> {os.path.join(a.root, 'funcnorm_analysis.json')}")

if __name__ == "__main__":
    main()
