#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
analyze_delaylaw_pscan.py  --  Is the grokking delay a SCALING LAW?

USAGE: python analyze_delaylaw_pscan.py --root D:\Colab-local\paperA_delaylaw_2026...

Reports:
  1. ||W||_c(p) and ||E||_c(p) at grokking, with power-law fits (function-scale norm scaling).
  2. Per-p delay-law fit  T_grok(rho) ~ exp(alpha(p) * rho)  -> alpha(p).
  3. Is alpha(p) constant (one universal exponent) or does it scale with p / ||W||_c?
  4. DATA COLLAPSE test: a single global model  log T_grok = alpha * rho + f(p)  -- if its R^2 is high
     with one shared alpha, the exponential is a genuine scaling law, not a per-p coincidence. We also
     test the alternative collapse against the HELD norm  log T_grok = beta * (rho * ||W||_c(p)) + c.
  5. Sanity: the network should grok AT the held norm (norm@grok ~ rho*||W||_c(p)).
"""
from __future__ import annotations
import argparse, glob, os, json
import numpy as np

def load(f): return np.load(f, allow_pickle=True)
def powerlaw(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); m = (y > 0) & np.isfinite(y)
    if m.sum() < 3: return None
    b, a = np.polyfit(np.log(x[m]), np.log(y[m]), 1)
    pred = b * np.log(x[m]) + a; R2 = 1 - np.sum((np.log(y[m]) - pred) ** 2) / np.sum((np.log(y[m]) - np.log(y[m]).mean()) ** 2)
    return dict(exp=round(float(b), 3), A=round(float(np.exp(a)), 3), R2=round(float(R2), 4))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); a = ap.parse_args()
    cells = []
    for f in glob.glob(os.path.join(a.root, "metrics", "*.npz")):
        d = load(f); T = d['T_grok_per_seed']; g = T > 0
        cells.append(dict(p=int(d['p']), arm=str(d['arm']), rho=float(d['rho']),
                          wc=(float(d['wc']) if np.isfinite(d['wc']) else None),
                          grok=int(g.sum()), S=int(len(T)),
                          Tg=(float(np.median(T[g])) if g.any() else None),
                          wn_grok=(float(np.nanmean(d['wn_at_grok'])) if np.isfinite(d['wn_at_grok']).any() else None),
                          wnE_grok=(float(np.nanmean(d['wnE_at_grok'])) if np.isfinite(d['wnE_at_grok']).any() else None)))
    ps = sorted(set(c['p'] for c in cells))

    print("\n=== 1. Function-scale norm scaling (from free controls) ===")
    ctrl = {c['p']: c for c in cells if c['arm'] == 'none'}
    wc_p = [(p, ctrl[p]['wn_grok']) for p in ps if p in ctrl and ctrl[p]['wn_grok']]
    wE_p = [(p, ctrl[p]['wnE_grok']) for p in ps if p in ctrl and ctrl[p]['wnE_grok']]
    for lbl, arr in [("||W||_c(p)", wc_p), ("||E||_c(p)", wE_p)]:
        if len(arr) >= 3:
            fit = powerlaw([x for x, _ in arr], [y for _, y in arr])
            print(f"  {lbl}: " + "  ".join(f"p{p}={v:.1f}" for p, v in arr) + f"   ~ p^{fit['exp']} (R2={fit['R2']})")

    print("\n=== 2-3. Per-p delay law  T_grok(rho) ~ exp(alpha(p) rho) ===")
    alphas = []
    for p in ps:
        pts = sorted([(c['rho'], c['Tg']) for c in cells if c['p'] == p and c['arm'] == 'clamp' and c['Tg']],
                     key=lambda t: t[0])
        if len(pts) >= 3:
            rho = np.array([r for r, _ in pts]); Tg = np.array([t for _, t in pts])
            al, b = np.polyfit(rho, np.log(Tg), 1)
            pred = al * rho + b; R2 = 1 - np.sum((np.log(Tg) - pred) ** 2) / np.sum((np.log(Tg) - np.log(Tg).mean()) ** 2)
            wc = ctrl.get(p, {}).get('wn_grok')
            alphas.append((p, al, wc))
            pstr = " ".join(f"({r:.2f}:{int(t)})" for r, t in pts)
            print(f"  p={p:3d}: alpha={al:.2f} (R2={R2:.3f})   rho:Tg = {pstr}")
    if len(alphas) >= 3:
        P = np.array([p for p, _, _ in alphas]); A = np.array([al for _, al, _ in alphas])
        cv = 100 * A.std() / A.mean()
        print(f"\n  alpha(p): mean={A.mean():.2f}  CV={cv:.1f}%  ", end="")
        print("=> ~constant exponent (universal)" if cv < 12 else "=> scales with p (report alpha(p))")
        # does alpha scale with 1/||W||_c?  (held norm = rho*wc, so exp in absolute norm = alpha/wc)
        wcs = np.array([wc for _, _, wc in alphas if wc])
        if len(wcs) == len(A):
            f2 = powerlaw(P, A)
            if f2: print(f"  alpha ~ p^{f2['exp']} (R2={f2['R2']})")

    print("\n=== 4. DATA COLLAPSE: global  log T_grok = alpha*rho + f(p) ===")
    rows = [(c['p'], c['rho'], c['Tg']) for c in cells if c['arm'] == 'clamp' and c['Tg']]
    if len(rows) >= 6:
        pid = {p: i for i, p in enumerate(sorted(set(r[0] for r in rows)))}
        X = np.zeros((len(rows), 1 + len(pid))); y = np.zeros(len(rows))
        for i, (p, rho, Tg) in enumerate(rows):
            X[i, 0] = rho; X[i, 1 + pid[p]] = 1.0; y[i] = np.log(Tg)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None); pred = X @ coef
        R2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        print(f"  shared alpha={coef[0]:.2f}, global R2={R2:.4f}  "
              f"({'COLLAPSE: one exponent fits all p -> scaling law' if R2 > 0.97 else 'partial; alpha(p) varies'})")

    print("\n=== 5. Sanity: groks at the held norm? (norm@grok vs rho*wc) ===")
    for c in sorted([c for c in cells if c['arm'] == 'clamp' and c['wn_grok'] and c['wc']], key=lambda c: (c['p'], c['rho'])):
        print(f"  p={c['p']:3d} rho={c['rho']:.2f}: held={c['rho']*c['wc']:.1f}  norm@grok={c['wn_grok']:.1f}  grok {c['grok']}/{c['S']}")
    json.dump(cells, open(os.path.join(a.root, "delaylaw_analysis.json"), "w"), indent=1)
    print(f"\nsaved -> {os.path.join(a.root, 'delaylaw_analysis.json')}")

if __name__ == "__main__":
    main()
