#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
analyze_denserho.py  --  Dense rho sweep at p=59: firm up the exponential delay law and locate its knee.

USAGE: python analyze_denserho.py --root D:\Colab-local\paperA_denserho_2026...

Reports:
  1. Per-rho median T_grok with bootstrap 95% CI, grok fraction, max/budget (censoring check).
  2. Bootstrap 95% CI on the exponent alpha, fit over the low-rho linear regime.
  3. Knee detection: largest contiguous low-rho window that stays log-linear (R^2 >= 0.995), and the
     point beyond which the curve goes sub-exponential (ratio of consecutive steps drops).
  4. Sanity: groks at the held norm.
"""
from __future__ import annotations
import argparse, glob, os, json
import numpy as np

def load(f): return np.load(f, allow_pickle=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True)
    ap.add_argument("--knee_r2", type=float, default=0.995); a = ap.parse_args()
    rng = np.random.default_rng(0)
    wc = None
    rows = []
    for f in sorted(glob.glob(os.path.join(a.root, "metrics", "*.npz"))):
        d = load(f); T = d['T_grok_per_seed'].astype(float); g = T > 0
        rho = float(d['rho']); arm = str(d['arm']); bud = int(d['max_steps'])
        if arm == "none":
            w = d['wn_at_grok']; w = w[np.isfinite(w)]; wc = float(np.median(w)) if len(w) else None
            continue
        if g.sum() < 3: 
            rows.append(dict(rho=rho, med=None, grok=int(g.sum()), S=len(T), bud=bud)); continue
        Tg = T[g]; med = float(np.median(Tg))
        boot = [np.median(rng.choice(Tg, len(Tg), replace=True)) for _ in range(3000)]
        rows.append(dict(rho=rho, med=med, lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)),
                         grok=int(g.sum()), S=len(T), mx=int(Tg.max()), bud=bud,
                         wn=float(np.nanmean(d['wn_at_grok']))))
    rows = sorted([r for r in rows], key=lambda r: r['rho'])
    print(f"\n||W||_c(p=59) measured = {wc:.2f}" if wc else "\n(control wc not found)")
    print("\n=== 1. Per-rho median [95% CI], grok, censoring ===")
    print("  rho   median  [   95% CI    ]  grok   max/budget")
    for r in rows:
        if r['med'] is None:
            print(f"  {r['rho']:.2f}   --     (grok {r['grok']}/{r['S']})"); continue
        cen = "" if r['mx'] < 0.95 * r['bud'] else "  <-CENSORED?"
        print(f"  {r['rho']:.2f}  {r['med']:7.0f} [{r['lo']:6.0f},{r['hi']:6.0f}]  {r['grok']:2d}/{r['S']:<2d}  {r['mx']}/{r['bud']}{cen}")

    pts = [(r['rho'], r['med'], r['lo'], r['hi']) for r in rows if r['med']]
    rho = np.array([x[0] for x in pts]); Tg = np.array([x[1] for x in pts])

    print("\n=== 2-3. Exponential fit + knee detection ===")
    def fitw(i, j):
        x, y = rho[i:j + 1], np.log(Tg[i:j + 1]); A, b = np.polyfit(x, y, 1)
        pred = A * x + b; R2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        return A, b, R2
    # global fit over all available rho
    Ag, bg, R2g = fitw(0, len(rho) - 1)
    boots = []
    for _ in range(3000):
        jj = np.sort(rng.choice(len(rho), len(rho), replace=True))
        if len(set(rho[jj])) < 2: continue
        boots.append(np.polyfit(rho[jj], np.log(Tg[jj]), 1)[0])
    glo, ghi = np.percentile(boots, [2.5, 97.5])
    print(f"  global fit over rho in [{rho[0]:.2f},{rho[-1]:.2f}] ({len(rho)} pts): "
          f"alpha={Ag:.2f} [95% CI {glo:.2f},{ghi:.2f}]  R^2={R2g:.4f}")
    # longest CONTIGUOUS window (any start/end) with R^2 >= threshold -> the clean exponential regime
    best = None
    for i in range(len(rho)):
        for j in range(i + 2, len(rho)):
            A, b, R2 = fitw(i, j)
            if R2 >= a.knee_r2 and (best is None or (j - i) > (best[1] - best[0])):
                best = (i, j, A, b, R2)
    if best:
        i, j, A, b, R2 = best
        print(f"  cleanest exponential window: rho in [{rho[i]:.2f},{rho[j]:.2f}] "
              f"({j - i + 1} pts)  alpha={A:.2f}  R^2={R2:.4f}")
        for r0, t0 in zip(rho, Tg):
            if r0 > rho[j] + 1e-9:
                pred = np.exp(A * r0 + b)
                print(f"    rho={r0:.2f}: actual {t0:.0f} vs extrapolation {pred:.0f}  (ratio {t0/pred:.2f}) -> saturation")
            elif r0 < rho[i] - 1e-9:
                print(f"    rho={r0:.2f}: below the clean window (low-norm curvature / floor regime)")
    else:
        print(f"  (no window reached R^2>={a.knee_r2}; rely on the global fit above)")

    print("\n=== 4. Sanity: groks at held norm ===")
    if wc:
        for r in rows:
            if r['med']: print(f"  rho={r['rho']:.2f}: held={r['rho']*wc:.1f}  norm@grok={r['wn']:.1f}")
    json.dump(rows, open(os.path.join(a.root, "denserho_analysis.json"), "w"), indent=1, default=float)

if __name__ == "__main__":
    main()
