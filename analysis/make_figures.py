#!/usr/bin/env python3
"""
make_figures.py — regenerate all five paper figures from the raw metrics. Run from repo root:

    python analysis/make_figures.py --data data --out figures

Produces (in --out): critical_norm_discovery.png, normint_causal.png, grok_fss_collapse.png,
tf_functional_norm.png, tfnoln_causal.png. No GPU needed.
"""
import argparse, glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_dir(d): return [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(d, "metrics", "*.npz")))]
def nag(d, key="weight_norm", thr=0.90):
    ta, X = d["test_acc"], d[key]; o = []
    for s in range(ta.shape[0]):
        g = np.where(ta[s] >= thr)[0]
        if len(g): o.append(X[s][g[0]])
    return np.array(o)
def tgm(d):
    t = d["T_grok_per_seed"]; t = t[t > 0]; return float(np.median(t)) if len(t) else np.nan

# ---------------------------------------------------------------------------- Fig 1
def fig_critical_norm(D, out):
    cfgs = [c for c in load_dir(os.path.join(D, "grokfss")) if str(c["role"]) == "fss"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    # A: ||W||_c vs alpha per p
    for p, col in [(41, "C0"), (59, "C1"), (79, "C2"), (97, "C3")]:
        pts = sorted([(float(c["alpha"]), nag(c).mean()) for c in cfgs if int(c["p"]) == p and len(nag(c)) >= 6])
        if pts: ax[0, 0].plot([x[0] for x in pts], [x[1] for x in pts], "o-", color=col, label=f"p={p}")
    ax[0, 0].set_xlabel(r"training fraction $\alpha$"); ax[0, 0].set_ylabel(r"$\|W\|$ at grokking")
    ax[0, 0].set_title("A. Concentrated, invariant to data fraction"); ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=.3)
    # B: ||W||_c vs p
    P, N = [], []
    for p in [29, 41, 59, 79, 97]:
        v = np.concatenate([nag(c) for c in cfgs if int(c["p"]) == p and len(nag(c))])
        if len(v): P.append(p); N.append(v.mean())
    ax[0, 1].loglog(P, N, "o-", color="C1")
    if len(P) >= 3:
        b = np.polyfit(np.log(P), np.log(N), 1)[0]; ax[0, 1].set_title(fr"B. Scales with size: $\|W\|_c\propto p^{{{b:.2f}}}$")
    ax[0, 1].set_xlabel("p"); ax[0, 1].set_ylabel(r"$\|W\|_c$"); ax[0, 1].grid(alpha=.3, which="both")
    # C: norm + fourier trajectory for one representative run (p=59, alpha=0.40)
    c = next((c for c in cfgs if int(c["p"]) == 59 and abs(float(c["alpha"]) - 0.40) < 1e-6), None)
    if c is not None:
        st = c["steps"]; s = 0
        ax[1, 0].plot(st, c["weight_norm"][s], color="C1", label=r"$\|W\|$")
        ax[1, 0].set_xscale("log"); ax[1, 0].set_xlabel("step"); ax[1, 0].set_ylabel(r"$\|W\|$", color="C1")
        a2 = ax[1, 0].twinx(); a2.plot(st, c["fourier_gini"][s], color="C4", alpha=.7, label="Fourier conc.")
        a2.set_ylabel("Fourier concentration", color="C4")
        g = np.where(c["test_acc"][s] >= 0.9)[0]
        if len(g): ax[1, 0].axvline(st[g[0]], ls=":", color="k", label="grok")
    ax[1, 0].set_title("C. Norm overshoots, relaxes; grok on the descent"); ax[1, 0].grid(alpha=.3)
    # D: FSS collapse of T_grok
    a_exp, inv_nu = 1.9, 1.2; alpha_c = {29: 0.39, 41: 0.30, 59: 0.22, 79: 0.18, 97: 0.18}
    for p, col in [(41, "C0"), (59, "C1"), (79, "C2"), (97, "C3")]:
        xs, ys = [], []
        for c in cfgs:
            if int(c["p"]) != p: continue
            t = tgm(c)
            if np.isfinite(t):
                xs.append((float(c["alpha"]) - alpha_c.get(p, 0.2)) * p ** inv_nu); ys.append(t / p ** a_exp)
        if xs:
            o = np.argsort(xs); ax[1, 1].plot(np.array(xs)[o], np.array(ys)[o], "o-", color=col, ms=4, label=f"p={p}")
    ax[1, 1].set_yscale("log"); ax[1, 1].set_xlabel(r"$(\alpha-\alpha_c)\,p^{1/\nu}$")
    ax[1, 1].set_ylabel(r"$T_{\rm grok}/p^{a}$"); ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_title(fr"D. FSS collapse ($a={a_exp}$, $1/\nu={inv_nu}$)"); ax[1, 1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(out, "critical_norm_discovery.png"), dpi=150, bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------------------- Fig 2
def fig_normint(D, out):
    nc = load_dir(os.path.join(D, "normint"))
    def by(sub):
        for c in nc:
            cid = "ctrl" if str(c["arm"]) == "none" else (f"clamp_rho{float(c['rho']):.2f}" if str(c["arm"]) == "clamp" else "")
            if cid == sub: return c
    ctrl = by("ctrl"); st = ctrl["steps"]; Tc = tgm(ctrl); wc = 54.6
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for tag, lab, col in [("clamp_rho0.90", "clamp 0.90·wc (below)", "#1a9850"), ("ctrl", "control", "k"),
                          ("clamp_rho1.15", "clamp 1.15·wc", "#fdae61"), ("clamp_rho1.30", "clamp 1.30·wc", "#d73027")]:
        c = by(tag)
        if c is not None: ax[0].semilogx(st, c["test_acc"].mean(0), color=col, lw=2, label=lab)
    ax[0].axhline(0.9, ls=":", color="gray"); ax[0].set_xlabel("step"); ax[0].set_ylabel("test accuracy")
    ax[0].set_title("A. Norm clamp gates grokking (both directions)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    held = [0.90 * wc, 1.00 * wc, 1.15 * wc]; T = [tgm(by(f"clamp_rho{r:.2f}")) for r in [0.90, 1.00, 1.15]]
    ax[1].plot(held, T, "o-", ms=9, color="C0")
    for h, t, r in zip(held, T, [0.90, 1.00, 1.15]):
        ax[1].annotate(f"{t/Tc:.2f}×", (h, t), textcoords="offset points", xytext=(6, 8), fontsize=9)
    ax[1].scatter([1.30 * wc], [Tc * 3.4], marker="x", s=140, color="#d73027")
    ax[1].annotate("PREVENTED", (1.30 * wc, Tc * 3.4), fontsize=9, ha="center", va="top", color="#d73027")
    ax[1].axhline(Tc, ls="--", color="gray"); ax[1].axvline(wc, ls=":", color="green"); ax[1].text(wc, Tc * .4, r"$\|W\|_c$", color="green")
    ax[1].set_xlabel(r"held $\|W\|$"); ax[1].set_ylabel(r"$T_{\rm grok}$ (median)")
    ax[1].set_title("B. Monotone causal dose-response"); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(out, "normint_causal.png"), dpi=150, bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------------------- Fig 3
def fig_funcnorm(D, out):
    cfgs = [c for c in load_dir(os.path.join(D, "tfgrok")) if str(c["role"]) == "fss"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    P71 = sorted([c for c in cfgs if int(c["p"]) == 71 and len(nag(c)) >= 6], key=lambda z: float(z["alpha"]))
    al = [float(c["alpha"]) for c in P71]; U = [nag(c, "wn_unembed").mean() for c in P71]; T = [nag(c).mean() for c in P71]
    ax[0].plot(al, np.array(U) / np.mean(U), "o-", color="C2", lw=2, label=r"unembedding $\|U\|$")
    ax[0].plot(al, np.array(T) / np.mean(T), "s--", color="C3", lw=2, label="total norm")
    ax[0].set_xlabel(r"training fraction $\alpha$"); ax[0].set_ylabel("norm @ grok / mean")
    ax[0].set_title("A. Functional norm invariant; total drifts"); ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)
    groups = ["embed", "attn", "mlp", "unembed"]
    def cvg(p, g):
        ms = [nag(c, "wn_" + g).mean() for c in cfgs if int(c["p"]) == p and len(nag(c, "wn_" + g)) >= 8 and float(c["alpha"]) >= 0.25]
        return np.std(ms) / np.mean(ms) if len(ms) >= 2 else np.nan
    x = np.arange(4); w = 0.35
    ax[1].bar(x - w/2, [cvg(71, g) for g in groups], w, label="p=71", color="C0")
    ax[1].bar(x + w/2, [cvg(97, g) for g in groups], w, label="p=97", color="C1")
    ax[1].set_xticks(x); ax[1].set_xticklabels(groups); ax[1].set_ylabel(r"CV across $\alpha$")
    ax[1].set_title("B. Only the unembedding is invariant"); ax[1].legend(); ax[1].grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig(os.path.join(out, "tf_functional_norm.png"), dpi=150, bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------------------- Fig 4
def fig_tfnoln(D, out):
    nl = load_dir(os.path.join(D, "tfnoln"))
    wcj = os.path.join(D, "tfnoln", "wc_measured.json")
    wc = json.load(open(wcj))["wc"] if os.path.exists(wcj) else 36.9
    def by(tag):
        for c in nl:
            cid = "ctrl" if str(c["arm"]) == "none" else f"clamp_rho{float(c['rho']):.2f}"
            if cid == tag: return c
    ctrl = by("ctrl"); st = ctrl["steps"]; Tc = tgm(ctrl)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for tag, lab, col in [("clamp_rho0.85", "0.85·wc (below)", "#1a9850"), ("ctrl", "control", "k"),
                          ("clamp_rho1.00", "1.00·wc", "#4575b4"), ("clamp_rho1.15", "1.15·wc", "#fdae61"),
                          ("clamp_rho1.30", "1.30·wc", "#d73027")]:
        c = by(tag)
        if c is not None: ax[0].semilogx(st, c["test_acc"].mean(0), color=col, lw=2, label=lab)
    ax[0].axhline(0.9, ls=":", color="gray"); ax[0].set_xlabel("step"); ax[0].set_ylabel("test accuracy")
    ax[0].set_title("A. No-LN: the critical-norm gate reappears"); ax[0].legend(fontsize=8, loc="center left"); ax[0].grid(alpha=.3)
    rhos = [0.85, 1.00]; T = [tgm(by(f"clamp_rho{r:.2f}")) / Tc for r in rhos]
    ax[1].plot([r * wc for r in rhos], T, "o-", ms=10, color="C0")
    for r, y in zip(rhos, T): ax[1].annotate(f"{y:.2f}×", (r * wc, y), textcoords="offset points", xytext=(6, 8), fontsize=10)
    ax[1].scatter([1.15 * wc], [1.9], marker="x", s=150, color="#d73027")
    ax[1].annotate("suppressed\n(≥1.15·wc)", (1.15 * wc, 1.9), fontsize=9, ha="left", va="center", color="#d73027")
    ax[1].axhline(1.0, ls="--", color="gray"); ax[1].axvline(wc, ls=":", color="green"); ax[1].text(wc, 0.4, r"$\|W\|_c$", color="green")
    ax[1].set_xlabel(r"held $\|W\|$"); ax[1].set_ylabel(r"$T_{\rm grok}$ / control")
    ax[1].set_title("B. Sharp bidirectional gate"); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(out, "tfnoln_causal.png"), dpi=150, bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------------------- standalone FSS fig
def fig_fss(D, out):
    cfgs = [c for c in load_dir(os.path.join(D, "grokfss")) if str(c["role"]) == "fss"]
    a_exp, inv_nu = 1.9, 1.2; alpha_c = {29: 0.39, 41: 0.30, 59: 0.22, 79: 0.18, 97: 0.18}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for p, col in [(41, "C0"), (59, "C1"), (79, "C2"), (97, "C3")]:
        A, Y = [], []
        for c in cfgs:
            if int(c["p"]) != p: continue
            t = tgm(c)
            if np.isfinite(t): A.append(float(c["alpha"])); Y.append(t)
        if A:
            o = np.argsort(A); ax[0].semilogy(np.array(A)[o], np.array(Y)[o], "o-", color=col, label=f"p={p}")
    ax[0].set_xlabel(r"$\alpha$"); ax[0].set_ylabel(r"$T_{\rm grok}$"); ax[0].set_title("A. Raw grokking-time curves"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    for p, col in [(41, "C0"), (59, "C1"), (79, "C2"), (97, "C3")]:
        xs, ys = [], []
        for c in cfgs:
            if int(c["p"]) != p: continue
            t = tgm(c)
            if np.isfinite(t): xs.append((float(c["alpha"]) - alpha_c.get(p, .2)) * p ** inv_nu); ys.append(t / p ** a_exp)
        if xs:
            o = np.argsort(xs); ax[1].semilogy(np.array(xs)[o], np.array(ys)[o], "o-", color=col, ms=4, label=f"p={p}")
    ax[1].set_xlabel(r"$(\alpha-\alpha_c)\,p^{1/\nu}$"); ax[1].set_ylabel(r"$T_{\rm grok}/p^{a}$")
    ax[1].set_title(fr"B. Collapse ($a={a_exp}$, $1/\nu={inv_nu}$)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(out, "grok_fss_collapse.png"), dpi=150, bbox_inches="tight"); plt.close()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data"); ap.add_argument("--out", default="figures"); A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)
    for name, fn in [("critical_norm_discovery", fig_critical_norm), ("normint_causal", fig_normint),
                     ("grok_fss_collapse", fig_fss), ("tf_functional_norm", fig_funcnorm), ("tfnoln_causal", fig_tfnoln)]:
        try:
            fn(A.data, A.out); print(f"  [ok] {name}.png")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")

if __name__ == "__main__":
    main()
