#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
run_grokfss.py  --  Canonical modular-arithmetic GROKKING runner with finite-size
scaling, for Paper A's headline empirics (the regime where delayed generalization is
UNAMBIGUOUS). Companion to run_critfss.py (the analytic linear model). Same critical-
phenomena toolkit, same resumable/timestamped storage.

WHY THIS TASK
-------------
Isotropic linear regression shows only ~1% epoch-wise double-descent recovery (verified on
the existing data) -- too weak to convincingly carry the saddle-node / delta=1/2 claims.
Modular addition (a+b) mod p with weight decay groks dramatically (train acc -> 100% early,
test acc -> 100% much later: Power et al. 2022; Nanda et al. 2023; Liu et al. Omnigrok 2023),
so the transition, its critical point, and its exponents are cleanly measurable.

CRITICAL-PHENOMENON DESIGN
--------------------------
  control parameter : alpha = train fraction         -> critical alpha_c (grokking phase transition)
  finite-size axis  : p (modulus); dataset size = p^2 -> FSS:  T_grok(alpha,p) = p^a G((alpha-alpha_c) p^b)
  optimizer signature: sweep r (learning rate) -> beta ; sweep lambda (weight decay) -> alpha-exponent
  fluctuation       : many seeds (different train/test split per seed) -> per-seed T_grok distribution

MODEL (2-layer MLP; batched over the seed axis S on GPU via autograd + manual AdamW)
  embed a,b with shared E in R^{p x d};  x = [e_a ; e_b] in R^{2d}
  h = gelu(x W1 + b1),  W1 in R^{2d x H};  logits = h W2 + b2,  W2 in R^{H x p}
  loss = cross-entropy over p classes;  decoupled weight decay on {E,W1,W2} (not biases).
  Per-seed independent train/test SPLIT of the p^2 pairs (so seed variance includes the split).
  Full-batch gradient descent on the train split (standard for grokking).

METRICS (logged per seed per log-step) -- definitions:
  train_loss/test_loss : mean cross-entropy on the train/test split
  train_acc/test_acc   : top-1 accuracy on train/test split
  weight_norm          : sqrt(||E||^2+||W1||^2+||W2||^2)   (Omnigrok control variable)
  spec_entropy_E       : spectral entropy of E   (= -sum p_i ln p_i, p_i=sg_i^2/sum sg^2)
  spec_entropy_W2      : spectral entropy of readout W2
  eff_rank_E           : exp(spec_entropy_E)
  fourier_gini         : Gini of the embedding's Fourier power over token frequencies
                         (LOW pre-grok = spread; HIGH post-grok = concentrated on few freqs;
                          the mechanistic "specialization" order parameter, Nanda et al.)

DETECTION (stored; re-doable offline from full curves):
  T_mem  = first log-step with train_acc >= acc_mem   (default 0.99)
  T_grok = first log-step with test_acc  >= acc_grok  (default 0.90); -1 if never (censored)
  grok_delay = T_grok - T_mem
We store full per-seed accuracy/loss/observable curves so any threshold can be re-applied.

STORAGE / RESUME (identical scheme to run_critfss.py; kill-safe, per-config resume):
  root = <--root>\paperA_grokfss_<UTC>\  with  grid_spec.json, checkpoint.json,
  metrics\<config_id>.npz, Temp\<config_id>_at_grok.npz, results_summary.json, run.log.
  Atomic writes (tmp + os.replace); relaunch with --resume <folder> continues where it stopped.

USAGE (PowerShell):
  python run_grokfss.py --dry-run
  python run_grokfss.py --device cuda --root D:\Colab-local
  python run_grokfss.py --resume D:\Colab-local\paperA_grokfss_2026....
  python run_grokfss.py --quick            # tiny smoke test (small p, few steps)
"""
from __future__ import annotations
import argparse, json, math, os, time, hashlib, datetime, platform
import numpy as np
try:
    import torch
    import torch.nn.functional as F
except Exception:                                              # pragma: no cover
    print("ERROR: PyTorch (CUDA build) required."); raise

# =====================================================================================
# grid
# =====================================================================================
def default_grid():
    optimizers = ["adamw"]                 # SGD groks unreliably; add "sgd" for the beta contrast if desired
    primes = [29, 41, 59, 79, 97]          # finite-size axis (dataset = p^2)
    alphas = [0.30, 0.35, 0.40, 0.50, 0.65]  # control parameter; critical alpha_c sits in here
    d_model, H = 128, 256
    lam_ref, lr_ref = 1.0, 1e-3            # Nanda-style grokking defaults
    lams = [0.3, 1.0, 3.0]                 # for alpha-exponent
    lrs  = [3e-4, 1e-3, 3e-3]              # for beta
    seeds = 20
    max_steps = 60_000

    grid = []
    # (A) FSS / alpha_c core: vary p x alpha at reference (lam,lr)
    for opt in optimizers:
        for p in primes:
            for al in alphas:
                grid.append(dict(role="fss", optim=opt, p=p, alpha=al, d_model=d_model, H=H,
                                 lam=lam_ref, lr=lr_ref, seeds=seeds, max_steps=max_steps))
    # (B) optimizer-signature core: vary lam x lr at reference (p, alpha)
    for opt in optimizers:
        for lam in lams:
            for lr in lrs:
                grid.append(dict(role="ab", optim=opt, p=59, alpha=0.5, d_model=d_model, H=H,
                                 lam=lam, lr=lr, seeds=seeds, max_steps=max_steps))
    # (A2) LOW-ALPHA EXTENSION: bracket alpha_c at large p (where alpha_c < 0.30), so the FSS collapse
    #      and the independent fluctuation-based alpha_c can be completed. These alphas are NEW values,
    #      so config_id is new and a --resume of the original run executes ONLY these 9 configs.
    #      Longer budget (100k) so alpha near alpha_c has a fair chance to grok (divergence), while the
    #      lowest alphas may stay censored -- that censoring is the lower bracket on alpha_c (the signal).
    for opt in optimizers:
        for p in [59, 79, 97]:
            for al in [0.15, 0.20, 0.25]:
                grid.append(dict(role="fss", optim=opt, p=p, alpha=al, d_model=d_model, H=H,
                                 lam=lam_ref, lr=lr_ref, seeds=seeds, max_steps=100_000))
    seen, uniq = set(), []
    for c in grid:
        cid = config_id(c)
        if cid not in seen:
            seen.add(cid); uniq.append(c)
    return uniq

def quick_grid():
    return [dict(role="fss", optim="adamw", p=23, alpha=0.5, d_model=64, H=128,
                 lam=1.0, lr=3e-3, seeds=4, max_steps=8000),
            dict(role="fss", optim="adamw", p=23, alpha=0.4, d_model=64, H=128,
                 lam=1.0, lr=3e-3, seeds=4, max_steps=8000)]

def config_id(c):
    s = (f"{c['optim']}_p{c['p']}_a{c['alpha']:.2f}_d{c['d_model']}_H{c['H']}"
         f"_wd{c['lam']:.0e}_lr{c['lr']:.0e}")
    return s.replace("+", "")

# =====================================================================================
# spectral / mechanistic observables (batched over seed axis S)
# =====================================================================================
def spec_entropy(M):
    """M: (S, m, n) -> (S,) spectral entropy of singular values."""
    sg = torch.linalg.svdvals(M)
    p = sg.pow(2); p = p / (p.sum(-1, keepdim=True) + 1e-30)
    return -(p * (p + 1e-30).log()).sum(-1)

def fourier_gini(E):
    """E: (S, p, d) embedding. Gini of token-frequency power (mechanistic specialization).
       FFT over the token axis p; power per freq = sum_d |.|^2; Gini in [0,1], rises at grok."""
    S, p, d = E.shape
    F_ = torch.fft.rfft(E, dim=1)                      # (S, p//2+1, d) complex
    power = (F_.abs() ** 2).sum(-1)                    # (S, F)
    power = power[:, 1:]                               # drop DC
    power = power / (power.sum(-1, keepdim=True) + 1e-30)
    sorted_p, _ = torch.sort(power, dim=-1)            # ascending
    n = sorted_p.shape[-1]
    idx = torch.arange(1, n + 1, device=E.device, dtype=E.dtype)
    # Gini = (2*sum(i*x_i)/ (n*sum x) ) - (n+1)/n
    gini = (2 * (idx * sorted_p).sum(-1)) / (n * sorted_p.sum(-1) + 1e-30) - (n + 1) / n
    return gini                                        # (S,)

def log_step_grid(max_steps, n_points=360):
    pts = np.unique(np.round(np.geomspace(1, max_steps, n_points)).astype(int))
    return np.unique(np.concatenate([[0], np.clip(pts, 0, max_steps)]))

# =====================================================================================
# train one config (all S seeds batched on GPU via autograd; manual AdamW)
# =====================================================================================
def run_config(c, device, dtype=torch.float32, acc_mem=0.99, acc_grok=0.90):
    p, S = c["p"], c["seeds"]
    d, H = c["d_model"], c["H"]
    lam, lr, opt = c["lam"], c["lr"], c["optim"]
    cid = config_id(c)
    base_seed = int(hashlib.sha1(cid.encode()).hexdigest(), 16) % (2**31)
    g = torch.Generator(device=device).manual_seed(base_seed)

    # full dataset of all pairs
    a_all = torch.arange(p, device=device).repeat_interleave(p)     # (p^2,)
    b_all = torch.arange(p, device=device).repeat(p)                # (p^2,)
    y_all = (a_all + b_all) % p                                     # labels
    M = p * p
    n_tr = max(2, int(round(c["alpha"] * M)))
    n_te = M - n_tr
    if n_te < 1:
        n_tr = M - 1; n_te = 1

    # per-seed train/test split (different per seed)
    perm = torch.stack([torch.randperm(M, generator=g, device=device) for _ in range(S)])  # (S,M)
    tr_idx = perm[:, :n_tr]; te_idx = perm[:, n_tr:]
    a_tr = a_all[tr_idx]; b_tr = b_all[tr_idx]; y_tr = y_all[tr_idx]   # (S, n_tr)
    a_te = a_all[te_idx]; b_te = b_all[te_idx]; y_te = y_all[te_idx]   # (S, n_te)

    # parameters with leading seed axis; init ~ standard
    def param(*shape, scale):
        t = (scale * torch.randn(S, *shape, generator=g, device=device, dtype=dtype))
        return t.requires_grad_(True)
    E  = param(p, d,   scale=1.0 / math.sqrt(d))
    W1 = param(2 * d, H, scale=1.0 / math.sqrt(2 * d))
    b1 = param(H,      scale=0.0 + 1e-6)
    W2 = param(H, p,   scale=1.0 / math.sqrt(H))
    b2 = param(p,      scale=0.0 + 1e-6)
    params = [E, W1, b1, W2, b2]
    decay_mask = [True, True, False, True, False]                  # no wd on biases
    if opt == "adamw":
        b1m, b2m, aeps = 0.9, 0.999, 1e-8
        ms = [torch.zeros_like(t) for t in params]
        vs = [torch.zeros_like(t) for t in params]

    def forward(a_idx, b_idx):
        ea = torch.gather(E, 1, a_idx.unsqueeze(-1).expand(S, a_idx.shape[1], d))  # (S,B,d)
        eb = torch.gather(E, 1, b_idx.unsqueeze(-1).expand(S, b_idx.shape[1], d))
        x = torch.cat([ea, eb], dim=-1)                                           # (S,B,2d)
        h = F.gelu(torch.einsum('sbe,seh->sbh', x, W1) + b1.unsqueeze(1))
        logits = torch.einsum('sbh,shp->sbp', h, W2) + b2.unsqueeze(1)            # (S,B,p)
        return logits

    def loss_acc(a_idx, b_idx, y):
        logits = forward(a_idx, b_idx)
        logp = F.log_softmax(logits, dim=-1)
        ce = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1).mean(dim=1)            # (S,)
        acc = (logits.argmax(-1) == y).float().mean(dim=1)                        # (S,)
        return ce, acc

    steps_to_log = set(int(s) for s in log_step_grid(c["max_steps"]))
    logged, rec = [], {k: [] for k in ['train_loss', 'test_loss', 'train_acc', 'test_acc',
                                       'weight_norm', 'spec_entropy_E', 'spec_entropy_W2',
                                       'eff_rank_E', 'fourier_gini']}
    t0 = time.time()
    for step in range(c["max_steps"] + 1):
        if step in steps_to_log:
            with torch.no_grad():
                ce_tr, acc_tr = loss_acc(a_tr, b_tr, y_tr)
                ce_te, acc_te = loss_acc(a_te, b_te, y_te)
                wn = torch.sqrt(sum(t.pow(2).sum(dim=tuple(range(1, t.ndim))) for t in [E, W1, W2]))
                seE = spec_entropy(E); seW2 = spec_entropy(W2); fg = fourier_gini(E)
            logged.append(step)
            for k, val in [('train_loss', ce_tr), ('test_loss', ce_te), ('train_acc', acc_tr),
                           ('test_acc', acc_te), ('weight_norm', wn), ('spec_entropy_E', seE),
                           ('spec_entropy_W2', seW2), ('eff_rank_E', seE.exp()), ('fourier_gini', fg)]:
                rec[k].append(val.cpu().numpy())
        if step == c["max_steps"]:
            break
        # full-batch gradient step (sum over seeds -> per-seed grads via autograd)
        ce_tr, _ = loss_acc(a_tr, b_tr, y_tr)
        loss = ce_tr.sum()
        for t in params:
            if t.grad is not None: t.grad = None
        loss.backward()
        with torch.no_grad():
            if opt == "adamw":
                t_ = step + 1
                for i, t in enumerate(params):
                    if decay_mask[i]:
                        t.mul_(1 - lr * lam)                       # decoupled weight decay
                    ms[i].mul_(b1m).add_(t.grad, alpha=1 - b1m)
                    vs[i].mul_(b2m).addcmul_(t.grad, t.grad, value=1 - b2m)
                    mhat = ms[i] / (1 - b1m ** t_); vhat = vs[i] / (1 - b2m ** t_)
                    t.addcdiv_(mhat, vhat.sqrt().add_(aeps), value=-lr)
            else:  # sgd + coupled L2
                for i, t in enumerate(params):
                    gr = t.grad + (lam * t if decay_mask[i] else 0.0)
                    t.add_(gr, alpha=-lr)

    steps = np.asarray(logged, np.int64)
    out = {k: np.stack(v, axis=1).astype(np.float32) for k, v in rec.items()}      # (S,T)
    # detection
    tr_acc = out['train_acc']; te_acc = out['test_acc']
    Tmem = np.full(S, -1, np.int64); Tgrok = np.full(S, -1, np.int64)
    for s in range(S):
        im = np.where(tr_acc[s] >= acc_mem)[0]
        if len(im): Tmem[s] = steps[im[0]]
        ig = np.where(te_acc[s] >= acc_grok)[0]
        if len(ig): Tgrok[s] = steps[ig[0]]
    delay = np.where((Tgrok > 0) & (Tmem >= 0), Tgrok - Tmem, -1).astype(np.int64)
    payload = dict(steps=steps, p=np.int64(p), alpha=np.float32(c["alpha"]),
                   d_model=np.int64(d), H=np.int64(H), lam=np.float32(lam), lr=np.float32(lr),
                   optim=str(opt), role=str(c["role"]), max_steps=np.int64(c["max_steps"]),
                   n_train=np.int64(n_tr), n_test=np.int64(n_te),
                   acc_mem=np.float32(acc_mem), acc_grok=np.float32(acc_grok),
                   T_mem_per_seed=Tmem, T_grok_per_seed=Tgrok, grok_delay_per_seed=delay, **out)
    # snapshot at grok (rich, for later analysis)
    snap = {}
    for s in range(S):
        j = int(np.argmin(np.abs(steps - Tgrok[s]))) if Tgrok[s] > 0 else -1
        for k in out:
            snap.setdefault(k, []).append(out[k][s, j] if j >= 0 else np.nan)
    snap = {k: np.asarray(v, np.float32) for k, v in snap.items()}
    return payload, snap, time.time() - t0, (S, n_tr, p)

# =====================================================================================
# atomic IO / checkpoint (identical scheme to run_critfss.py)
# =====================================================================================
def atomic_write_json(path, obj):
    tmp = path + ".tmp"; json.dump(obj, open(tmp, "w"), indent=1); os.replace(tmp, path)

def atomic_savez(path, **arrays):
    tmp = path + ".tmp.npz"; np.savez_compressed(tmp, **arrays); os.replace(tmp, path)

def load_checkpoint(root):
    p = os.path.join(root, "checkpoint.json")
    try:
        return json.load(open(p)) if os.path.exists(p) else {}
    except Exception:
        return {}

def log_line(root, msg):
    line = f"[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] {msg}"
    print(line, flush=True)
    open(os.path.join(root, "run.log"), "a").write(line + "\n")

# =====================================================================================
# main
# =====================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\Colab-local")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    grid = quick_grid() if a.quick else default_grid()
    print(f"configs: {len(grid)} | device: {a.device}")
    if a.dry_run:
        for c in grid[:15]:
            print(f"  {config_id(c)}  role={c['role']} p={c['p']} alpha={c['alpha']} "
                  f"seeds={c['seeds']} steps={c['max_steps']}")
        print(f"  ... ({len(grid)} total)")
        return

    if a.resume:
        root = a.resume; assert os.path.isdir(root), f"--resume not found: {root}"
    else:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        root = os.path.join(a.root, f"paperA_grokfss_{ts}")
    os.makedirs(os.path.join(root, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(root, "Temp"), exist_ok=True)
    atomic_write_json(os.path.join(root, "grid_spec.json"),
                      [dict(config_id=config_id(c), **c) for c in grid])
    ckpt = load_checkpoint(root)
    log_line(root, f"start: {len(grid)} configs, device={a.device}, host={platform.node()}, "
                   f"resume={'yes' if a.resume else 'no'}, done={sum(1 for v in ckpt.values() if v.get('done'))}")
    device = torch.device(a.device)
    spath = os.path.join(root, "results_summary.json")
    summary = json.load(open(spath)) if os.path.exists(spath) else []

    for i, c in enumerate(grid):
        cid = config_id(c)
        if ckpt.get(cid, {}).get("done"):
            continue
        try:
            payload, snap, wall, dims = run_config(c, device)
        except RuntimeError as e:
            log_line(root, f"[{i+1}/{len(grid)}] {cid} FAILED: {e}"); continue
        atomic_savez(os.path.join(root, "metrics", cid + ".npz"), **payload)
        atomic_savez(os.path.join(root, "Temp", cid + "_at_grok.npz"), **snap)
        tg = payload["T_grok_per_seed"]; gd = payload["grok_delay_per_seed"]
        conv = tg[tg > 0]; gdc = gd[gd > 0]
        rowsum = dict(config_id=cid, optim=c["optim"], role=c["role"], p=c["p"], alpha=c["alpha"],
                      lam=c["lam"], lr=c["lr"], seeds=int(dims[0]), n_train=int(dims[1]),
                      n_grokked=int((tg > 0).sum()),
                      T_grok_median=float(np.median(conv)) if len(conv) else None,
                      grok_delay_median=float(np.median(gdc)) if len(gdc) else None,
                      T_grok_cv=float(conv.std() / conv.mean()) if len(conv) else None,
                      wall_s=round(wall, 2))
        summary.append(rowsum); atomic_write_json(spath, summary)
        ckpt[cid] = {"done": True, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     "n_grokked": rowsum["n_grokked"], "wall_s": round(wall, 2)}
        atomic_write_json(os.path.join(root, "checkpoint.json"), ckpt)
        log_line(root, f"[{i+1}/{len(grid)}] {cid}  grok={rowsum['n_grokked']}/{dims[0]}  "
                       f"Tgrok={rowsum['T_grok_median']}  delay={rowsum['grok_delay_median']}  {wall:.1f}s")
    log_line(root, f"DONE. results in: {root}")
    print(f"\nAll outputs under: {root}")

if __name__ == "__main__":
    main()
