#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
analyze_tfnoln_long.py  --  Did the un-normalized transformer's high-rho "suppression" resolve into a DELAY?

USAGE: python analyze_tfnoln_long.py --root D:\Colab-local\paperA_tfnoln_long_2026...

Reports, per clamp cell:
  * grok fraction and median T_grok at the extended budget,
  * whether the previously-"suppressed" rho>=1.15 now crosses 0.9 (=> delay, not prevention),
  * final test accuracy and end-slope (rising => still delaying, would grok with more budget).
And fits the second-architecture delay law  T_grok(rho) ~ exp(alpha*rho)  on the lr=5e-4 cells,
for comparison with the MLP (alpha ~ 7.5).
"""
from __future__ import annotations
import argparse, glob, os, json
import numpy as np

def load(f): return np.load(f, allow_pickle=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(a.root, "metrics", "*clamp_rho*.npz"))):
        d = load(f); st = d['steps']; te = d['test_acc'].mean(0); tr = d['train_acc'].mean(0)
        T = d['T_grok_per_seed']; g = T > 0
        third = int(len(st) * 0.66)
        slope = (te[-1] - te[third]) / (st[-1] - st[third] + 1) * 1000
        rows.append(dict(rho=float(d['rho']), lr=float(d['lr']), budget=int(d['max_steps']),
                         S=int(len(T)), grok=int(g.sum()),
                         Tg=(float(np.median(T[g])) if g.any() else None),
                         test_end=float(te[-1]), test_max=float(te.max()),
                         slope_per1k=float(slope),
                         diverged=bool(d['diverged']) if 'diverged' in d.files else False))
    print("\n=== Extended-budget un-normalized transformer: delay or prevention? ===")
    print("  rho   lr     budget   grok    T_grok   test_end  end-slope/1k  verdict")
    for r in sorted(rows, key=lambda r: (r['lr'], r['rho'])):
        if r['grok'] > 0 and r['test_max'] >= 0.9:
            verdict = "GROKKED -> DELAY (caveat closed)"
        elif r['slope_per1k'] > 0.002:
            verdict = "still rising -> delay (needs more budget)"
        elif r['diverged']:
            verdict = "DIVERGED (uninformative)"
        else:
            verdict = "flat -> possible true prevention"
        tg = f"{r['Tg']:.0f}" if r['Tg'] else "  -  "
        print(f"  {r['rho']:.2f}  {r['lr']:.0e}  {r['budget']:>7d}  {r['grok']:2d}/{r['S']:<2d}  {tg:>7s}  "
              f"{r['test_end']:.2f}     {r['slope_per1k']:+.4f}     {verdict}")

    # second-architecture delay law on lr=5e-4 cells that grokked
    fit_cells = [r for r in rows if abs(r['lr'] - 5e-4) < 1e-9 and r['Tg']]
    if len(fit_cells) >= 3:
        rho = np.array([r['rho'] for r in fit_cells]); Tg = np.array([r['Tg'] for r in fit_cells])
        a_, b_ = np.polyfit(rho, np.log(Tg), 1)
        pred = a_ * rho + b_; R2 = 1 - np.sum((np.log(Tg) - pred) ** 2) / np.sum((np.log(Tg) - np.log(Tg).mean()) ** 2)
        print(f"\n=== Second-architecture delay law (lr=5e-4): T_grok ~ exp({a_:.2f}*rho), R2={R2:.3f} ===")
        print(f"    (MLP reference: alpha ~ 7.5.  alpha2 ~2x the MLP => the FORM (exponential, grok-at-held-norm, saturation) is cross-architectural; the RATE is architecture-specific.)")
    else:
        print("\n  (need >=3 grokked lr=5e-4 cells to fit the delay law; extend budget on the slow ones.)")
    json.dump(rows, open(os.path.join(a.root, "tfnoln_long_analysis.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
