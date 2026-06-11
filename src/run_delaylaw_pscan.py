#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
run_delaylaw_pscan.py  --  Turn the grokking DELAY LAW from a single-p curve into a SCALING LAW.

WHAT WE ALREADY HAVE (run_efflr_dissoc, p=59):
  Holding the weight norm at rho*||W||_c delays grokking as  T_grok ~ exp(alpha*rho)  (R^2=0.99),
  the network groks at the held norm, and the norm dominates the learning rate.

WHAT THIS ADDS (the step that makes it a law, not a curve):
  Measure  T_grok(p, rho)  across several task sizes p, so we can:
    (1) fit  alpha(p)  and ask whether the delay exponent is constant, or scales with p / ||W||_c(p);
    (2) test a DATA COLLAPSE -- whether  log T_grok  for all p falls on one curve after rescaling rho by
        a p-dependent scale (e.g. against rho, or against the held norm rho*||W||_c(p)) -- which is the
        standard evidence that an exponential fit is a genuine scaling law rather than a per-p coincidence;
    (3) compare the measured exponent to the first-passage / norm-separation delay theory, providing a
        direct causal validation of that theory rather than a correlational one.

DESIGN
  * For each p we FIRST run a free control (arm=none) to MEASURE ||W||_c(p) = median weight norm at
    grokking (persisted to wc_measured.json, keyed by p); the clamp arms then hold ||W||=rho*||W||_c(p).
    (Self-contained: ||W||_c is measured, not assumed.)
  * Matched counterfactual: the per-seed init and data split derive from base_key=(p,alpha,d,H,lam) ONLY,
    so every arm at a given p starts identically; only (arm,rho) differ. config_id (for storage/resume)
    includes p/arm/rho/lr; the seed does not.
  * Continuous clamp re-projects ||W|| to rho*||W||_c every step after t_int, WITHOUT resetting the AdamW
    moments (a reset spikes the first post-reset step and re-grows the norm, washing out the intervention).
  * Rich logging: full train/test curves, total AND per-layer norms (E, W1, W2), spectral entropy, Fourier
    gini, plus norm-at-grok per layer -- so the same run also gives ||W||_c(p) and the per-layer
    (function-scale) norm scaling for free.
  * Adaptive budget: T_grok grows with both p and rho, so max_steps scales with both; the expensive
    rho=1.30 cells are run only at the two best-anchored primes. config_id excludes max_steps, so an
    interrupted run resumes cleanly; cheap cells (low rho, all p) already determine alpha(p).

USAGE (PowerShell):
  python run_delaylaw_pscan.py --dry-run
  python run_delaylaw_pscan.py --quick --device cuda --out_root D:\Colab-local
  python run_delaylaw_pscan.py --device cuda --out_root D:\Colab-local
  python run_delaylaw_pscan.py --resume D:\Colab-local\paperA_delaylaw_2026...
  python analyze_delaylaw_pscan.py --root D:\Colab-local\paperA_delaylaw_2026...
"""
import argparse, json, math, os, time, hashlib, datetime, platform
import numpy as np
try:
    import torch
    import torch.nn.functional as F
except Exception:
    print("ERROR: PyTorch required."); raise

# ---------------- grid ----------------
def budget_for(p, rho):
    base = {0.85: 15_000, 1.00: 25_000, 1.15: 80_000, 1.30: 160_000}[rho]
    return int(round(base * max(1.0, p / 59.0) / 1000.0) * 1000)

def default_grid():
    d, H = 128, 256
    primes = [31, 41, 53, 59, 71]
    seeds, t_int, lr, lam, alpha = 12, 500, 1e-3, 1.0, 0.40
    def mk(p, arm, rho):
        ms = budget_for(p, 1.00 if arm == "none" else rho)
        return dict(optim="adamw", p=p, alpha=alpha, d_model=d, H=H, lam=lam, lr=lr,
                    seeds=seeds, t_int=t_int, arm=arm, rho=float(rho), max_steps=ms)
    grid = []
    for p in primes:
        grid.append(mk(p, "none", 1.00))                       # control FIRST -> measures ||W||_c(p)
        for rho in (0.85, 1.00, 1.15):                          # cheap cells at every p -> alpha(p)
            grid.append(mk(p, "clamp", rho))
    for p in (41, 59):                                          # expensive high-rho only at anchors
        grid.append(mk(p, "clamp", 1.30))
    return dedupe(grid)

def quick_grid():
    d, H = 32, 64
    def mk(p, arm, rho, ms):
        return dict(optim="adamw", p=p, alpha=0.40, d_model=d, H=H, lam=1.0, lr=3e-3,
                    seeds=3, t_int=150, arm=arm, rho=float(rho), max_steps=ms)
    return dedupe([mk(23, "none", 1.00, 4000), mk(23, "clamp", 0.85, 4000), mk(23, "clamp", 1.15, 6000),
                   mk(29, "none", 1.00, 4000), mk(29, "clamp", 1.15, 6000)])

def base_key(c):  # matched init+data per p (NOT lr/arm/rho)
    return f"{c['p']}_{c['alpha']:.2f}_{c['d_model']}_{c['H']}_{c['lam']:.0e}".replace("+", "")
def config_id(c):
    tag = "ctrl" if c["arm"] == "none" else f"clamp_rho{c['rho']:.2f}"
    return (f"adamw_p{c['p']}_a{c['alpha']:.2f}_d{c['d_model']}_H{c['H']}"
            f"_wd{c['lam']:.0e}_lr{c['lr']:.0e}_{tag}").replace("+", "")
def dedupe(grid):
    seen, u = set(), []
    for c in grid:
        k = config_id(c)
        if k not in seen: seen.add(k); u.append(c)
    return u

# ---------------- observables ----------------
def spec_entropy(M):
    sg = torch.linalg.svdvals(M); pw = sg.pow(2); pw = pw / (pw.sum(-1, keepdim=True) + 1e-30)
    return -(pw * (pw + 1e-30).log()).sum(-1)
def fourier_gini(E):
    S, p, d = E.shape
    Fc = torch.fft.rfft(E, dim=1); power = (Fc.abs() ** 2).sum(-1)[:, 1:]
    power = power / (power.sum(-1, keepdim=True) + 1e-30)
    sp, _ = torch.sort(power, dim=-1); n = sp.shape[-1]
    idx = torch.arange(1, n + 1, device=E.device, dtype=E.dtype)
    return (2 * (idx * sp).sum(-1)) / (n * sp.sum(-1) + 1e-30) - (n + 1) / n
def log_step_grid(max_steps, n_points=360):
    pts = np.unique(np.round(np.geomspace(1, max_steps, n_points)).astype(int))
    return np.unique(np.concatenate([[0], np.clip(pts, 0, max_steps)]))
def frob(t):
    return torch.sqrt(t.pow(2).sum(dim=tuple(range(1, t.ndim))))

# ---------------- one config ----------------
def run_config(c, device, wc=None, dtype=torch.float32, acc_mem=0.99, acc_grok=0.90):
    p, S, d, H = c["p"], c["seeds"], c["d_model"], c["H"]
    lam, lr, rho, arm, t_int = c["lam"], c["lr"], c["rho"], c["arm"], c["t_int"]
    base_seed = int(hashlib.sha1(base_key(c).encode()).hexdigest(), 16) % (2**31)
    g = torch.Generator(device=device).manual_seed(base_seed)

    a_all = torch.arange(p, device=device).repeat_interleave(p)
    b_all = torch.arange(p, device=device).repeat(p)
    y_all = (a_all + b_all) % p
    M = p * p; n_tr = max(2, int(round(c["alpha"] * M))); n_te = max(1, M - n_tr)
    perm = torch.stack([torch.randperm(M, generator=g, device=device) for _ in range(S)])
    tr, te = perm[:, :n_tr], perm[:, n_tr:n_tr + n_te]
    a_tr, b_tr, y_tr = a_all[tr], b_all[tr], y_all[tr]
    a_te, b_te, y_te = a_all[te], b_all[te], y_all[te]

    def par(*shape, scale):
        return (scale * torch.randn(S, *shape, generator=g, device=device, dtype=dtype)).requires_grad_(True)
    E  = par(p, d,    scale=1.0 / math.sqrt(d))
    W1 = par(2 * d, H, scale=1.0 / math.sqrt(2 * d))
    b1 = par(H,       scale=1e-6)
    W2 = par(H, p,    scale=1.0 / math.sqrt(H))
    b2 = par(p,       scale=1e-6)
    params = [E, W1, b1, W2, b2]; decay = [True, True, False, True, False]
    wmats = {"E": E, "W1": W1, "W2": W2}
    b1m, b2m, aeps = 0.9, 0.999, 1e-8
    ms = [torch.zeros_like(t) for t in params]; vs = [torch.zeros_like(t) for t in params]

    def total_norm():
        return torch.sqrt(sum(wmats[k].pow(2).sum(dim=tuple(range(1, wmats[k].ndim))) for k in wmats))
    def gnorm(name):
        return torch.sqrt(wmats[name].pow(2).sum(dim=tuple(range(1, wmats[name].ndim))))
    def forward(ai, bi):
        ea = torch.gather(E, 1, ai.unsqueeze(-1).expand(S, ai.shape[1], d))
        eb = torch.gather(E, 1, bi.unsqueeze(-1).expand(S, bi.shape[1], d))
        x = torch.cat([ea, eb], dim=-1)
        h = F.gelu(torch.einsum('sbe,seh->sbh', x, W1) + b1.unsqueeze(1))
        return torch.einsum('sbh,shp->sbp', h, W2) + b2.unsqueeze(1)
    def loss_acc(ai, bi, y):
        lg = forward(ai, bi); lp = F.log_softmax(lg, -1)
        ce = -lp.gather(-1, y.unsqueeze(-1)).squeeze(-1).mean(1)
        acc = (lg.argmax(-1) == y).float().mean(1)
        return ce, acc
    def clamp_total(target):
        sc = target / (total_norm() + 1e-30)
        for k in wmats:
            sh = [S] + [1] * (wmats[k].ndim - 1); wmats[k].mul_(sc.view(*sh))

    steps_to_log = set(int(s) for s in log_step_grid(c["max_steps"]))
    ckpt_steps = sorted(set(int(s) for s in np.unique(np.round(
        np.geomspace(1, c["max_steps"], 6)).astype(int))) | {0, t_int})
    keys = ['train_loss', 'test_loss', 'train_acc', 'test_acc', 'weight_norm',
            'wn_E', 'wn_W1', 'wn_W2', 'spec_entropy_E', 'eff_rank_E', 'fourier_gini']
    logged, rec = [], {k: [] for k in keys}
    E_ck, E_ck_steps = [], []
    nbi = nai = None
    for step in range(c["max_steps"] + 1):
        if step in steps_to_log:
            with torch.no_grad():
                ce_tr, acc_tr = loss_acc(a_tr, b_tr, y_tr)
                ce_te, acc_te = loss_acc(a_te, b_te, y_te)
                seE = spec_entropy(E)
                vals = [('train_loss', ce_tr), ('test_loss', ce_te), ('train_acc', acc_tr),
                        ('test_acc', acc_te), ('weight_norm', total_norm()),
                        ('wn_E', gnorm("E")), ('wn_W1', gnorm("W1")), ('wn_W2', gnorm("W2")),
                        ('spec_entropy_E', seE), ('eff_rank_E', seE.exp()), ('fourier_gini', fourier_gini(E))]
            logged.append(step)
            for k, v in vals: rec[k].append(v.cpu().numpy())
        if step in ckpt_steps:
            E_ck.append(E.detach().cpu().numpy().astype(np.float32)); E_ck_steps.append(step)
        if step == c["max_steps"]: break
        ce_tr, _ = loss_acc(a_tr, b_tr, y_tr); loss = ce_tr.sum()
        for t in params: t.grad = None
        loss.backward()
        with torch.no_grad():
            t_ = step + 1
            for i, t in enumerate(params):
                if decay[i]: t.mul_(1 - lr * lam)
                ms[i].mul_(b1m).add_(t.grad, alpha=1 - b1m)
                vs[i].mul_(b2m).addcmul_(t.grad, t.grad, value=1 - b2m)
                mhat = ms[i] / (1 - b1m ** t_); vhat = vs[i] / (1 - b2m ** t_)
                t.addcdiv_(mhat, vhat.sqrt().add_(aeps), value=-lr)
            if arm == "clamp" and wc is not None and step + 1 >= t_int:
                if step + 1 == t_int: nbi = total_norm().cpu().numpy()
                clamp_total(rho * wc)
                if step + 1 == t_int: nai = total_norm().cpu().numpy()

    steps = np.asarray(logged, np.int64)
    out = {k: np.stack(v, 1).astype(np.float32) for k, v in rec.items()}
    tr_acc, te_acc = out['train_acc'], out['test_acc']
    Tmem = np.full(S, -1, np.int64); Tgrok = np.full(S, -1, np.int64)
    for s in range(S):
        im = np.where(tr_acc[s] >= acc_mem)[0]; ig = np.where(te_acc[s] >= acc_grok)[0]
        if len(im): Tmem[s] = steps[im[0]]
        if len(ig): Tgrok[s] = steps[ig[0]]
    delay = np.where((Tgrok > 0) & (Tmem >= 0), Tgrok - Tmem, -1).astype(np.int64)
    def at_grok(key):
        v = []
        for s in range(S):
            if Tgrok[s] > 0:
                v.append(out[key][s, int(np.argmin(np.abs(steps - Tgrok[s])))])
            else:
                v.append(np.nan)
        return np.asarray(v, np.float32)
    payload = dict(steps=steps, p=np.int64(p), alpha=np.float32(c["alpha"]), d_model=np.int64(d),
                   H=np.int64(H), lam=np.float32(lam), lr=np.float32(lr), arm=str(arm),
                   rho=np.float32(rho), wc=np.float32(wc if wc is not None else np.nan),
                   t_int=np.int64(t_int), max_steps=np.int64(c["max_steps"]),
                   n_train=np.int64(n_tr), n_test=np.int64(n_te),
                   T_mem_per_seed=Tmem, T_grok_per_seed=Tgrok, grok_delay_per_seed=delay,
                   wn_at_grok=at_grok('weight_norm'), wnE_at_grok=at_grok('wn_E'),
                   wnW1_at_grok=at_grok('wn_W1'), wnW2_at_grok=at_grok('wn_W2'),
                   norm_before_int=(nbi if nbi is not None else np.full(S, np.nan, np.float32)),
                   norm_after_int=(nai if nai is not None else np.full(S, np.nan, np.float32)), **out)
    snap = dict(E_ckpts=np.stack(E_ck, 0), E_ckpt_steps=np.asarray(E_ck_steps, np.int64),
                T_grok=Tgrok, T_mem=Tmem, arm=str(arm), rho=np.float32(rho),
                wc=np.float32(wc if wc is not None else np.nan), p=np.int64(p))
    return payload, snap, time.time(), (S, n_tr)

# ---------------- atomic IO ----------------
def atomic_write_json(path, obj):
    tmp = path + ".tmp"; json.dump(obj, open(tmp, "w"), indent=1); os.replace(tmp, path)
def atomic_savez(path, **a):
    tmp = path + ".tmp.npz"; np.savez_compressed(tmp, **a); os.replace(tmp, path)
def load_json(path, default):
    try: return json.load(open(path)) if os.path.exists(path) else default
    except Exception: return default
def log_line(root, msg):
    line = f"[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] {msg}"
    print(line, flush=True); open(os.path.join(root, "run.log"), "a").write(line + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=r"D:\Colab-local")
    ap.add_argument("--resume", default=None); ap.add_argument("--device", default="cuda")
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    grid = quick_grid() if args.quick else default_grid()
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    root = args.resume or os.path.join(args.out_root,
            f"paperA_delaylaw_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(os.path.join(root, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(root, "Temp"), exist_ok=True)
    print(f"configs: {len(grid)} | device: {device}")
    if args.dry_run:
        for c in grid:
            print(f"   {config_id(c):42s} p={c['p']:3d} {c['arm']:5s} rho={c['rho']:.2f} "
                  f"steps={c['max_steps']:>7d}  seed_key={base_key(c)}")
        return
    atomic_write_json(os.path.join(root, "grid_spec.json"), [dict(config_id=config_id(c), **c) for c in grid])
    ckpt = load_json(os.path.join(root, "checkpoint.json"), {})
    wcm = load_json(os.path.join(root, "wc_measured.json"), {})    # {str(p): wc}
    log_line(root, f"start: {len(grid)} configs, device={device}, host={platform.node()}, "
                   f"resume={'yes' if args.resume else 'no'}, done={sum(1 for v in ckpt.values() if v.get('done'))}")
    spath = os.path.join(root, "results_summary.json"); summary = load_json(spath, [])
    for i, c in enumerate(grid):
        cid = config_id(c)
        if ckpt.get(cid, {}).get("done"): continue
        wc = wcm.get(str(c["p"])) if c["arm"] == "clamp" else None
        if c["arm"] == "clamp" and wc is None:
            log_line(root, f"[{i+1}/{len(grid)}] SKIP {cid}: ||W||_c(p={c['p']}) not measured yet (run its control first)")
            continue
        t0 = time.time(); payload, snap, _, (S, n_tr) = run_config(c, device, wc=wc); dt = time.time() - t0
        atomic_savez(os.path.join(root, "metrics", cid + ".npz"), **payload)
        atomic_savez(os.path.join(root, "Temp", cid + "_struct.npz"), **snap)
        tg = payload["T_grok_per_seed"]; conv = tg[tg > 0]
        if c["arm"] == "none":
            w = payload["wn_at_grok"]; w = w[np.isfinite(w)]
            if len(w):
                wcm[str(c["p"])] = float(np.median(w))
                atomic_write_json(os.path.join(root, "wc_measured.json"), wcm)
                log_line(root, f"   measured ||W||_c(p={c['p']}) = {wcm[str(c['p'])]:.2f} ({len(w)} grokked seeds)")
        row = dict(config_id=cid, p=c["p"], arm=c["arm"], rho=c["rho"], lr=c["lr"], seeds=S,
                   wc_used=(wc if wc is not None else None), budget=c["max_steps"],
                   grok_frac=float((tg > 0).mean()),
                   T_grok_median=(float(np.median(conv)) if len(conv) else None),
                   wn_at_grok=(float(np.nanmean(payload["wn_at_grok"])) if np.isfinite(payload["wn_at_grok"]).any() else None),
                   wnE_at_grok=(float(np.nanmean(payload["wnE_at_grok"])) if np.isfinite(payload["wnE_at_grok"]).any() else None),
                   seconds=round(dt, 1))
        summary = [r for r in summary if r.get("config_id") != cid] + [row]
        atomic_write_json(spath, summary)
        ckpt[cid] = dict(done=True, seconds=round(dt, 1)); atomic_write_json(os.path.join(root, "checkpoint.json"), ckpt)
        log_line(root, f"[{i+1}/{len(grid)}] {cid}  grok={int((tg>0).sum())}/{S}  "
                       f"Tg={row['T_grok_median']}  ||W||@grok={row['wn_at_grok']}  {dt:.1f}s")
    log_line(root, "ALL DONE")

if __name__ == "__main__":
    main()
