#!/usr/bin/env python3
"""
reproduce_numbers.py — recompute every headline number in the paper directly from the raw metrics,
and print it next to the value claimed in the text. Run from the repo root:

    python analysis/reproduce_numbers.py --data data

No GPU needed; reads the saved .npz trajectories only. Each block maps to a paper section.
"""
import argparse, glob, json, os
import numpy as np

def load_dir(d):
    return [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(d, "metrics", "*.npz")))]

def norm_at_grok(d, key="weight_norm", thr=0.90):
    """||.|| of `key` at the first step where test acc >= thr, per seed."""
    ta = d["test_acc"]; X = d[key]; st = d["steps"]; out = []
    for s in range(ta.shape[0]):
        g = np.where(ta[s] >= thr)[0]
        if len(g): out.append(X[s][g[0]])
    return np.array(out)

def Tgrok_median(d, thr=0.90):
    tg = d["T_grok_per_seed"]; tg = tg[tg > 0]
    return float(np.median(tg)) if len(tg) else np.nan

def hr(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data"); A = ap.parse_args()
    D = A.data

    # ---------------------------------------------------------------- Section 4: critical norm (MLP)
    hr("Section 4  —  The Critical Weight Norm (MLP, run_grokfss)")
    cfgs = load_dir(os.path.join(D, "grokfss"))
    def fss(role="fss"): return [c for c in cfgs if str(c["role"]) == role]
    print("  [claim] ||W||_c concentrated across alpha, CV 1-2%  (p=59):")
    for p in [59]:
        for c in sorted([c for c in fss() if int(c["p"]) == p], key=lambda z: float(z["alpha"])):
            v = norm_at_grok(c)
            if len(v) >= 6:
                print(f"      alpha={float(c['alpha']):.2f}: ||W||_c={v.mean():6.2f}  CV={v.std()/v.mean():.3f}")
    print("  [claim] threshold-robustness: ||W||_c changes <2% as grok thr 0.8->0.95  (p=59):")
    for thr in [0.80, 0.90, 0.95]:
        v = np.concatenate([norm_at_grok(c, thr=thr) for c in fss() if int(c["p"]) == 59 and len(norm_at_grok(c, thr=thr))])
        print(f"      thr={thr}: ||W||_c={v.mean():.2f}")
    print("  [claim] ||W||_c ~ p^0.38 :")
    P, N = [], []
    for p in [29, 41, 59, 79, 97]:
        v = np.concatenate([norm_at_grok(c) for c in fss() if int(c["p"]) == p and len(norm_at_grok(c))])
        if len(v): P.append(p); N.append(v.mean()); print(f"      p={p}: ||W||_c={v.mean():.1f}")
    if len(P) >= 3:
        b = np.polyfit(np.log(P), np.log(N), 1)[0]; print(f"      fitted exponent = p^{b:.2f}   (paper: 0.38)")

    # ---------------------------------------------------------------- Section 5: causal MLP
    hr("Section 5  —  Causal Test: clamp ||W|| (MLP, run_normint)  [Table 1]")
    nc = load_dir(os.path.join(D, "normint"))
    def get(tag): return next((c for c in nc if tag in str(c["arm"]) + f"_rho{float(c['rho']):.2f}_" + str(c.get("layer", ""))), None)
    def by_id(sub):
        for c in nc:
            cid = f"{str(c['arm'])}"
            if str(c["arm"]) == "clamp": cid = f"clamp_rho{float(c['rho']):.2f}"
            elif str(c["arm"]) == "none": cid = "ctrl"
            if cid == sub: return c
        return None
    ctrl = by_id("ctrl"); Tc = Tgrok_median(ctrl)
    print(f"  control: grok={int((ctrl['T_grok_per_seed']>0).sum())}/{ctrl['T_grok_per_seed'].size}  T_grok={Tc:.0f}")
    wc = 54.6
    for rho in [0.90, 1.00, 1.15, 1.30]:
        c = by_id(f"clamp_rho{rho:.2f}")
        if c is None: continue
        tg = Tgrok_median(c); gf = (c["T_grok_per_seed"] > 0).mean()
        rel = f"{tg/Tc:.2f}x" if np.isfinite(tg) else "PREVENTED"
        print(f"  clamp {rho:.2f}*wc (||W||={rho*wc:.1f}): grok={gf:.2f}  T_grok={tg if np.isfinite(tg) else 'None':>7}  ({rel})")
    print("  [paper: 0.90->0.34x, 1.00->0.73x, control 1.00x, 1.15->2.45x, 1.30->prevented]")

    # ---------------------------------------------------------------- Section 6: universality (LN transformer)
    hr("Section 6  —  Universality on the LayerNorm Transformer (run_tfgrok)")
    tg_ = load_dir(os.path.join(D, "tfgrok"))
    def tf_fss(): return [c for c in tg_ if str(c["role"]) == "fss"]
    print("  [claim] total norm DRIFTS but unembedding ||U|| is concentrated (CV across alpha):")
    for p in [71, 97]:
        def cv(key):
            ms = [norm_at_grok(c, key).mean() for c in tf_fss() if int(c["p"]) == p
                  and len(norm_at_grok(c, key)) >= 8 and float(c["alpha"]) >= 0.25]
            return np.std(ms) / np.mean(ms) if len(ms) >= 2 else np.nan
        print(f"      p={p}: CV(total)={cv('weight_norm'):.3f}  CV(||U||)={cv('wn_unembed'):.3f}")
    print("  [claim] ||U||_c ~ p^0.77 :")
    P, U = [], []
    for p in [53, 71, 97]:
        v = np.concatenate([norm_at_grok(c, "wn_unembed") for c in tf_fss() if int(c["p"]) == p and len(norm_at_grok(c, "wn_unembed"))])
        if len(v): P.append(p); U.append(v.mean()); print(f"      p={p}: ||U||_c={v.mean():.1f}")
    if len(P) >= 3:
        print(f"      fitted exponent = p^{np.polyfit(np.log(P), np.log(U), 1)[0]:.2f}   (paper: 0.77)")

    # ---------------------------------------------------------------- Section 6: differential clamp (LN)
    hr("Section 6  —  Differential clamp on the LayerNorm Transformer (run_tfnormint)")
    ni = load_dir(os.path.join(D, "tfnormint"))
    def tget(tag):
        for c in ni:
            arm = str(c["arm"]); grp = str(c.get("group", ""))
            cid = "ctrl" if arm == "none" else f"clamp_{grp}_rho{float(c['rho']):.2f}"
            if cid == tag: return c
        return None
    ct = tget("ctrl"); Tc2 = Tgrok_median(ct)
    print(f"  control: T_grok={Tc2:.0f}")
    for tag in ["clamp_unembed_rho0.85", "clamp_unembed_rho1.30", "clamp_mlp_rho0.85", "clamp_mlp_rho1.30"]:
        c = tget(tag)
        if c is None: continue
        tg = Tgrok_median(c); rel = f"{tg/Tc2:.2f}x" if np.isfinite(tg) else "no-grok"
        print(f"  {tag:26}: T_grok={tg if np.isfinite(tg) else 'None':>7} ({rel})")
    print("  [paper: clamping ||U|| and mlp move grok comparably -> NO clean single-norm gate under LN]")

    # ---------------------------------------------------------------- Section 6: no-LN causal gate
    hr("Section 6  —  No-LayerNorm Transformer: the gate REAPPEARS (run_tfnoln)")
    nl = load_dir(os.path.join(D, "tfnoln"))
    wcj = os.path.join(D, "tfnoln", "wc_measured.json")
    wc_nl = json.load(open(wcj))["wc"] if os.path.exists(wcj) else 36.9
    def nlget(tag):
        for c in nl:
            arm = str(c["arm"]); cid = "ctrl" if arm == "none" else f"clamp_rho{float(c['rho']):.2f}"
            if cid == tag: return c
        return None
    ctn = nlget("ctrl"); Tcn = Tgrok_median(ctn)
    print(f"  measured ||W||_c (from control) = {wc_nl:.1f}")
    print(f"  control: grok={int((ctn['T_grok_per_seed']>0).sum())}/{ctn['T_grok_per_seed'].size}  T_grok={Tcn:.0f}")
    for rho in [0.85, 1.00, 1.15, 1.30, 1.50]:
        c = nlget(f"clamp_rho{rho:.2f}")
        if c is None: continue
        tg = Tgrok_median(c); gf = (c["T_grok_per_seed"] > 0).mean()
        final_te = float(c["test_acc"].mean(0)[-1])
        rel = f"{tg/Tcn:.2f}x" if np.isfinite(tg) else f"PREVENTED (test_acc->{final_te:.2f})"
        print(f"  clamp {rho:.2f}*wc (||W||={rho*wc_nl:.1f}): grok={gf:.2f}  T_grok={tg if np.isfinite(tg) else 'None':>7}  ({rel})")
    print("  [paper: 0.85->0.16x (6x faster), 1.00->1.54x, >=1.15 suppressed below threshold]")

    print("\n" + "=" * 78 + "\nDone. Values above should match the paper's claims (small seed-noise aside).\n" + "=" * 78)

if __name__ == "__main__":
    main()
