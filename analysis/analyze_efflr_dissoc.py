#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
analyze_efflr_dissoc.py  --  Read-outs for run_efflr_dissoc.py.

USAGE: python analyze_efflr_dissoc.py --root D:\Colab-local\paperA_efflr_2026...

Produces (printed + saved to <root>\efflr_analysis.json):
  1. PREVENTION MAP: grok-fraction over the (rho x lr) clamp matrix. Read the boundary:
       vertical (depends on rho/NORM only) -> norm-state gates;
       diagonal (depends on lr too)        -> effective-LR matters.
  2. EFFECTIVE-LR CHECK: measured eff_lr_E across the matrix -> confirms raising lr boosts
       the effective LR (the lever works), and locates effective-LR-matched cells.
  3. DISSOCIATION: the knockout pairs -- a HIGH-norm cell with effective LR >= a grokking
       LOW-norm cell, that nevertheless does NOT grok. If such pairs exist, high effective LR
       does not cause grokking; the norm state gates it.
"""
from __future__ import annotations
import argparse, json, glob, os
import numpy as np

def load(f): return np.load(f, allow_pickle=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); a = ap.parse_args()
    cells = []
    for f in glob.glob(os.path.join(a.root, "metrics", "*.npz")):
        d = load(f)
        T = d['T_grok_per_seed']; g = (T > 0)
        cells.append(dict(arm=str(d['arm']), rho=float(d['rho']), lr=float(d['lr']),
                          wc=float(d['wc']), S=int(len(T)), grok=int(g.sum()),
                          grok_frac=float(g.mean()),
                          Tg=(float(np.median(T[g])) if g.any() else None),
                          effE=float(np.nanmean(d['eff_lr_E_post'])),
                          wn_grok=(float(np.nanmean([d['weight_norm'][s, np.argmin(np.abs(d['steps']-T[s]))]
                                   for s in range(len(T)) if g[s]])) if g.any() else None)))
    clamp = [c for c in cells if c['arm'] == 'clamp']
    rhos = sorted(set(c['rho'] for c in clamp)); lrs = sorted(set(c['lr'] for c in clamp))
    def cell(rho, lr): return next((c for c in clamp if c['rho'] == rho and c['lr'] == lr), None)

    print("\n=== 1. PREVENTION MAP: grok fraction over (rho x lr) ===")
    print("        " + "".join(f"lr={lr:.0e}".rjust(11) for lr in lrs))
    for rho in rhos:
        row = f"  rho={rho:.2f}"
        for lr in lrs:
            c = cell(rho, lr); row += (f"{c['grok']}/{c['S']}".rjust(11)) if c else "—".rjust(11)
        print(row)
    print("  (boundary vertical => set by rho/NORM; diagonal => effective-LR matters)")

    print("\n=== 2. EFFECTIVE-LR (measured eff_lr_E, post-intervention) over (rho x lr) ===")
    print("        " + "".join(f"lr={lr:.0e}".rjust(11) for lr in lrs))
    for rho in rhos:
        row = f"  rho={rho:.2f}"
        for lr in lrs:
            c = cell(rho, lr); row += (f"{c['effE']:.4f}".rjust(11)) if c else "—".rjust(11)
        print(row)
    print("  (along a row, eff LR should rise with lr => the lever works)")

    print("\n=== 3. DISSOCIATION — knockout pairs (high-norm prevented at eff LR >= a grokking low-norm cell) ===")
    grokking = [c for c in clamp if c['grok_frac'] >= 0.5]
    prevented = [c for c in clamp if c['grok_frac'] == 0.0]
    knockouts = []
    for pcell in prevented:
        for gcell in grokking:
            if pcell['rho'] > gcell['rho'] + 1e-9 and pcell['effE'] >= gcell['effE']:
                knockouts.append((pcell, gcell))
    if knockouts:
        for pc, gc in sorted(knockouts, key=lambda x: -x[0]['effE'])[:6]:
            print(f"  PREVENTED rho={pc['rho']:.2f} lr={pc['lr']:.0e} effLR={pc['effE']:.4f}  "
                  f">=  GROKKED rho={gc['rho']:.2f} lr={gc['lr']:.0e} effLR={gc['effE']:.4f} "
                  f"(Tg={gc['Tg']})  -> higher eff LR yet prevented: NORM gates.")
    else:
        print("  none found: in this matrix the prevented cells never reach a grokking cell's eff LR;")
        print("  inspect the prevention map directly (vertical vs diagonal boundary).")

    json.dump(dict(cells=cells, rhos=rhos, lrs=lrs,
                   knockouts=[(p['rho'], p['lr'], p['effE'], g['rho'], g['lr'], g['effE'])
                              for p, g in knockouts]),
              open(os.path.join(a.root, "efflr_analysis.json"), "w"), indent=1)
    print(f"\nsaved -> {os.path.join(a.root, 'efflr_analysis.json')}")

if __name__ == "__main__":
    main()
