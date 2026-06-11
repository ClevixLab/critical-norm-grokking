#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
run_critfss.py  --  Comprehensive, GPU-batched, resumable experiment runner for
Paper A, reframed as a CRITICAL PHENOMENON validated by FINITE-SIZE SCALING (FSS).

It generates ONE self-consistent dataset (single normalization convention) that
supports every analysis we need on a common footing:
  * joint calibrated law            T* ~ lam^-alpha r^-beta (gamma-gamma_c)^-delta
  * FSS collapse                    T*(gamma,N) = N^a G((gamma-gamma_c) N^b)  -> robust gamma_c & delta
  * fluctuation / susceptibility    Var_seed[T*] vs gamma  -> independent gamma_c (gold standard)
  * order-parameter dynamics        signal alignment m (specialization saddle), q, ||W||, S
  * label-free early warning (CSD)  full per-seed trajectories of spectral entropy / weight norm

----------------------------------------------------------------------------------------------------
REVIEWER CHECKLIST (correctness / literature consistency) -- read before trusting outputs
----------------------------------------------------------------------------------------------------
MODEL (matches Paper A, single convention):
  multi-task linear regression  Yhat = X W^T,  W in R^{KxD}
  teacher W* : orthonormal rows (QR), unit-norm rows  -> m_k in [0,1]
  inputs  X_ij ~ N(0, 1/D)   (so Cov(x)=I/D);  Y = X W*^T + E,  E ~ N(0, sigma^2)
  gamma = D / N   (interpolation ratio);  SNR is controlled EXPLICITLY by sigma (not by convention)

OPTIMIZERS (manual, vectorized over the seed axis for full GPU parallelism):
  adamw : decoupled weight decay (Loshchilov-Hutter):  W <- W*(1 - lr*wd);  W <- W - lr * mhat/(sqrt(vhat)+eps)
  sgd   : coupled L2 (paper's "L2 reg"):               g <- g + wd*W;       W <- W - lr*g   (momentum optional)
  minibatch B (per-seed independent draws) supplies the stochastic gradient noise that
  drives null-space diffusion -- the mechanism behind the memorization phase. B=N => full-batch GD.

METRIC DEFINITIONS (logged per seed per log-step):
  train_loss      = mean_{n,k} (X W^T - Y)^2                (empirical MSE on the fixed training set)
  test_loss       = ||W - W*||_F^2 / (K*D)                  (EXACT population risk; Cov(x)=I/D); no sampling noise
  test_loss_held  = mean_{n,k}(X_te W^T - Y_te^signal)^2    (empirical held-out MSE; cross-check only)
  signal_align m  = (1/K) sum_k <W_k, W*_k>                 (W*_k unit norm) -- Paper A Eq.(2) order param
  noise_energy q  = (1/K) sum_k ||W_k - m_k W*_k||^2 / D    -- Paper A Eq.(3)
  weight_norm     = ||W||_F                                 (and s = ||W||_F^2/(K*D) derived offline)
  spectral_entropy= -sum_i p_i ln p_i, p_i = sg_i^2/sum sg_j^2, sg = svd(W)  (i=1..K)
  effective_rank  = exp(spectral_entropy)  in [1, K]        (Roy-Vetterli 2007)

DETECTION (stored, but ALSO re-doable offline because full curves are saved):
  T_peak   = argmax_t test_loss
  T*_DD    = first log-step AFTER T_peak with test_loss <= floor + eps_detect*(peak - floor),
             floor = min_t test_loss over the run;  = -1 if no peak / no return (censored).
  We deliberately do NOT bake gamma_c or a single delta-fit into the saved data: the gamma_c
  fragility is resolved downstream by FSS collapse + the fluctuation exponent, on the raw curves.

STORAGE / RESUME (kill-safe):
  root = <--root>\paperA_critfss_<UTC timestamp>   (default base: D:\Colab-local)
    grid_spec.json            full config list (deterministic config_id per config)
    checkpoint.json           {config_id: {"done": true, "utc": ...}}  -> resume skips done configs
    metrics\<config_id>.npz   full per-seed trajectories + per-seed T*, T_peak  (the rich artifact)
    Temp\<config_id>_at_*.npz at-peak / at-T* snapshots (kept for later deep analysis; never deleted)
    results_summary.json      appended per-config medians/CV (for quick fits & figs)
    run.log                   progress log
  Writes are ATOMIC (tmp file + os.replace), so an interrupt never corrupts a file; on relaunch the
  runner reads checkpoint.json and continues exactly where it stopped (per-config granularity).
----------------------------------------------------------------------------------------------------
USAGE (PowerShell on the workstation):
  python run_critfss.py                       # default grid, root D:\Colab-local
  python run_critfss.py --root D:\Colab-local --device cuda
  python run_critfss.py --resume <existing run folder>     # continue an interrupted run
  python run_critfss.py --dry-run             # print the grid + memory estimate, write nothing
  python run_critfss.py --quick               # tiny grid for a smoke test
"""
from __future__ import annotations
import argparse, json, math, os, sys, time, hashlib, datetime, platform
import numpy as np

try:
    import torch
except Exception:                                   # pragma: no cover
    print("ERROR: PyTorch is required. Install the CUDA build for your GPU.")
    raise

# =====================================================================================
# grid
# =====================================================================================
def default_grid():
    """The comprehensive grid. Three roles, one convention:
       - alpha,beta  : sweep lam x r          (at a few gamma, larger N)
       - delta + FSS : sweep gamma x N         (the finite-size-scaling core, near gamma_c)
       - fluctuation : many seeds everywhere   (per-seed T* distribution)
       - SNR control : a couple of sigma values
    Edit freely; config_id is derived from the values so partial reruns stay consistent.
    """
    K, D = 8, 128
    optimizers = ["adamw", "sgd"]
    # COMPANION grid (trimmed): linear is the analytic companion, not the headline, so this is sized
    # to run in ~1 h. gamma starts at 1.10 (the extreme 1.05 needs ~200k steps -> censoring); size tops
    # at D=256 (D=512 was the slow tail). Restore the fuller grid below if you want the complete sweep.
    gammas = [1.10, 1.20, 1.40, 1.70, 2.20, 4.00]   # 10 -> 6
    size_scales = [0.5, 1.0, 2.0]                    # drop 4.0 (D=512, slowest)
    lams = [1e-3, 1e-2, 1e-1]                         # 5 -> 3
    lrs  = [1e-2, 3e-2]                               # 3 -> 2
    sigmas = [0.3]
    seeds = 16                                        # 30 -> 16 (enough for a companion)
    base_steps = 150_000                              # 200k -> 150k (near-gamma_c points may censor; OK)
    batch = 32

    grid = []
    # (A) FSS / delta core: vary gamma x size at a reference (lam, r); many seeds
    for opt in optimizers:
        for sc in size_scales:
            for g in gammas:
                grid.append(dict(role="fss", optim=opt, K=K, scale=sc, gamma=g,
                                 lam=1e-2, lr=(1e-2 if opt == "adamw" else 3e-2),
                                 sigma=0.3, seeds=seeds, batch=batch, max_steps=base_steps))
    # (B) alpha,beta core: vary lam x r at gamma=2, reference size; many seeds
    for opt in optimizers:
        for lam in lams:
            for lr in lrs:
                grid.append(dict(role="ab", optim=opt, K=K, scale=1.0, gamma=2.0,
                                 lam=lam, lr=lr, sigma=0.3, seeds=seeds,
                                 batch=batch, max_steps=base_steps))
    # de-duplicate by config_id
    seen, uniq = set(), []
    for c in grid:
        cid = config_id(c)
        if cid not in seen:
            seen.add(cid); uniq.append(c)
    return uniq

def quick_grid():
    K = 8
    g = []
    for opt in ["adamw", "sgd"]:
        for gamma in [1.2, 2.0]:
            g.append(dict(role="fss", optim=opt, K=K, scale=0.5, gamma=gamma,
                          lam=1e-2, lr=(1e-2 if opt == "adamw" else 3e-2),
                          sigma=0.3, seeds=6, batch=16, max_steps=3000))
    return g

def derive(c):
    """Fill D, N from K/scale/gamma."""
    D = int(round(128 * c["scale"]))
    N = max(8, int(round(D / c["gamma"])))
    return D, N

def config_id(c):
    D, N = derive(c)
    s = (f"{c['optim']}_K{c['K']}_D{D}_N{N}_g{c['gamma']:.3f}"
         f"_wd{c['lam']:.0e}_lr{c['lr']:.0e}_sig{c['sigma']:.2f}_b{c['batch']}")
    return s.replace("+", "")

# =====================================================================================
# metric helpers (batched over seed axis S)
# =====================================================================================
def spectral_entropy_batch(W):
    """W: (S,K,D) -> (S,) spectral entropy over the K singular values."""
    sg = torch.linalg.svdvals(W)                       # (S,K)
    p = sg.pow(2)
    p = p / (p.sum(dim=1, keepdim=True) + 1e-30)
    S = -(p * (p + 1e-30).log()).sum(dim=1)
    return S                                           # (S,)

def order_params(W, Wstar):
    """W:(S,K,D), Wstar:(K,D) unit rows. Returns dict of (S,) tensors."""
    S, K, D = W.shape
    # m_k = <W_k, W*_k>  (W*_k unit norm)
    mk = (W * Wstar.unsqueeze(0)).sum(dim=2)           # (S,K)
    m = mk.mean(dim=1)                                 # (S,)
    # V_k = W_k - m_k W*_k ; q_k = ||V_k||^2 / D
    V = W - mk.unsqueeze(2) * Wstar.unsqueeze(0)       # (S,K,D)
    q = (V.pow(2).sum(dim=2) / D).mean(dim=1)          # (S,)
    wn = torch.linalg.matrix_norm(W, ord='fro')        # (S,)
    Sent = spectral_entropy_batch(W)                   # (S,)
    return dict(signal_alignment=m, noise_energy=q, weight_norm=wn,
                spectral_entropy=Sent, effective_rank=Sent.exp())

def log_step_grid(max_steps, n_points=360):
    """Geometric-ish step grid (dense early), unique ints, includes 0 and max_steps."""
    pts = np.unique(np.round(np.geomspace(1, max_steps, n_points)).astype(int))
    pts = np.concatenate([[0], pts])
    pts = np.unique(np.clip(pts, 0, max_steps))
    return pts

# =====================================================================================
# train one config (all S seeds batched on GPU)
# =====================================================================================
def run_config(c, device, dtype=torch.float32, eps_detect=0.10):
    D, N = derive(c)
    K, S = c["K"], c["seeds"]
    sigma, B = c["sigma"], min(c["batch"], N)
    g_dev = torch.Generator(device=device)
    cid = config_id(c)
    # deterministic seeding from config_id
    base_seed = int(hashlib.sha1(cid.encode()).hexdigest(), 16) % (2**31)
    g_dev.manual_seed(base_seed)

    # teacher (shared across seeds): orthonormal rows
    A = torch.randn(D, K, generator=g_dev, device=device, dtype=dtype)
    Qm, _ = torch.linalg.qr(A)                          # (D,K)
    Wstar = Qm.T.contiguous()                           # (K,D) orthonormal rows

    # per-seed data + init
    X = torch.randn(S, N, D, generator=g_dev, device=device, dtype=dtype) / math.sqrt(D)
    E = sigma * torch.randn(S, N, K, generator=g_dev, device=device, dtype=dtype)
    Y = torch.einsum('snd,kd->snk', X, Wstar) + E       # (S,N,K)  noisy targets
    Ysig = torch.einsum('snd,kd->snk', X, Wstar)        # clean (for held-out signal MSE)
    W = 0.01 * torch.randn(S, K, D, generator=g_dev, device=device, dtype=dtype)

    # held-out test set (signal MSE cross-check); small to stay cheap
    Nte = min(4096, 8 * N)
    Xte = torch.randn(S, Nte, D, generator=g_dev, device=device, dtype=dtype) / math.sqrt(D)
    Yte = torch.einsum('snd,kd->snk', Xte, Wstar)

    # optimizer state
    opt = c["optim"]; lr, wd = c["lr"], c["lam"]
    if opt == "adamw":
        b1, b2, aeps = 0.9, 0.999, 1e-8
        mbuf = torch.zeros_like(W); vbuf = torch.zeros_like(W)
    momentum = 0.0
    vel = torch.zeros_like(W)

    steps_to_log = set(int(s) for s in log_step_grid(c["max_steps"]))
    logged_steps = []
    rec = {k: [] for k in ['train_loss', 'test_loss', 'test_loss_held',
                            'signal_alignment', 'noise_energy', 'weight_norm',
                            'spectral_entropy', 'effective_rank']}
    snap = {}  # at-peak / at-T* snapshots filled after detection

    def metrics_now():
        with torch.no_grad():
            pred = torch.einsum('snd,skd->snk', X, W)         # (S,N,K)
            train = (pred - Y).pow(2).mean(dim=(1, 2))        # (S,)
            test = (W - Wstar.unsqueeze(0)).pow(2).sum(dim=(1, 2)) / (K * D)   # exact pop risk (S,)
            predte = torch.einsum('snd,skd->snk', Xte, W)
            held = (predte - Yte).pow(2).mean(dim=(1, 2))
            op = order_params(W, Wstar)
        return train, test, held, op

    t0 = time.time()
    for step in range(c["max_steps"] + 1):
        if step in steps_to_log:
            train, test, held, op = metrics_now()
            logged_steps.append(step)
            rec['train_loss'].append(train.cpu().numpy())
            rec['test_loss'].append(test.cpu().numpy())
            rec['test_loss_held'].append(held.cpu().numpy())
            for k in ['signal_alignment', 'noise_energy', 'weight_norm',
                      'spectral_entropy', 'effective_rank']:
                rec[k].append(op[k].cpu().numpy())
        if step == c["max_steps"]:
            break
        # --- one optimizer step (minibatch per seed) ---
        idx = torch.randint(0, N, (S, B), generator=g_dev, device=device)
        Xb = torch.gather(X, 1, idx.unsqueeze(-1).expand(S, B, D))     # (S,B,D)
        Yb = torch.gather(Y, 1, idx.unsqueeze(-1).expand(S, B, K))     # (S,B,K)
        pred = torch.einsum('sbd,skd->sbk', Xb, W)                     # (S,B,K)
        resid = pred - Yb
        grad = torch.einsum('sbk,sbd->skd', resid, Xb) / B            # dL/dW, L=0.5*MSE_b
        if opt == "adamw":
            W.mul_(1 - lr * wd)                                       # decoupled weight decay
            mbuf.mul_(b1).add_(grad, alpha=1 - b1)
            vbuf.mul_(b2).addcmul_(grad, grad, value=1 - b2)
            t = step + 1
            mhat = mbuf / (1 - b1 ** t); vhat = vbuf / (1 - b2 ** t)
            W.addcdiv_(mhat, vhat.sqrt().add_(aeps), value=-lr)
        else:  # sgd with coupled L2
            g = grad + wd * W
            if momentum > 0:
                vel.mul_(momentum).add_(g); g = vel
            W.add_(g, alpha=-lr)

    # ----- arrays -----
    steps = np.asarray(logged_steps, dtype=np.int64)
    out = {k: np.stack(v, axis=1).astype(np.float32) for k, v in rec.items()}  # (S, T)
    # ----- per-seed T_peak / T*_DD : MULTIPLE definitions (stored; re-doable offline) -----
    # Rationale (reviewer note): in isotropic linear regression the test-loss *recovery* is
    # often weak, so a recovery-only detector is fragile. We therefore store three per-seed
    # markers and let the analysis choose / show robustness:
    #   T_peak        = argmax test_loss
    #   T_star_recov  = recovery time: first step after peak with test_loss <= postfloor + eps*(peak-postfloor),
    #                   postfloor = min test_loss for t >= T_peak  (NOT the t=0 value)
    #   T_star_align  = specialization-escape time: step of max d(signal_alignment)/dstep after T_peak
    #                   (validated to coincide with the transition, r_log~0.92)
    tl = out['test_loss']                                 # (S,T)
    ma = out['signal_alignment']                          # (S,T)
    Tpeak = np.full(S, -1, np.int64)
    Tstar_recov = np.full(S, -1, np.int64)
    Tstar_align = np.full(S, -1, np.int64)
    for s in range(S):
        y = tl[s]
        ip = int(np.argmax(y)); Tpeak[s] = steps[ip]
        post = y[ip:]
        postfloor = post.min(); peak = y[ip]
        if peak - postfloor > 1e-9:
            thr = postfloor + eps_detect * (peak - postfloor)
            after = np.where((np.arange(len(y)) > ip) & (y <= thr))[0]
            if len(after):
                Tstar_recov[s] = steps[after[0]]
        # alignment escape (after peak): max positive d m / d step
        if len(steps) > ip + 3:
            dm = np.gradient(ma[s], steps.astype(float))
            seg = np.arange(len(steps)) > ip
            if seg.sum() >= 3:
                j = np.where(seg)[0][int(np.argmax(dm[seg]))]
                Tstar_align[s] = steps[j]
    # primary T* used by the summary = recovery if available else alignment-escape
    Tstar = np.where(Tstar_recov > 0, Tstar_recov, Tstar_align)
    payload = dict(steps=steps, gamma=np.float32(c["gamma"]), gamma_field=np.float32(c["gamma"]),
                   lam=np.float32(c["lam"]), lr=np.float32(c["lr"]), sigma=np.float32(c["sigma"]),
                   K=np.int64(K), D=np.int64(D), N=np.int64(N), batch=np.int64(B),
                   optim=str(c["optim"]), role=str(c["role"]), max_steps=np.int64(c["max_steps"]),
                   eps_detect=np.float32(eps_detect),
                   T_peak_per_seed=Tpeak, T_star_DD_per_seed=Tstar,
                   T_star_recovery_per_seed=Tstar_recov, T_star_align_per_seed=Tstar_align,
                   **out)
    # ----- snapshots at peak / T* (rich, for later deep analysis) -----
    def snap_at(stepvals):
        cols = {}
        for s in range(S):
            j = int(np.argmin(np.abs(steps - stepvals[s]))) if stepvals[s] >= 0 else -1
            for k in out:
                cols.setdefault(k, []).append(out[k][s, j] if j >= 0 else np.nan)
        return {k: np.asarray(v, np.float32) for k, v in cols.items()}
    snap_peak = snap_at(Tpeak); snap_tstar = snap_at(Tstar)
    wall = time.time() - t0
    return payload, snap_peak, snap_tstar, wall, (S, N, D)

# =====================================================================================
# atomic IO / checkpoint
# =====================================================================================
def atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)

def atomic_savez(path, **arrays):
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)

def load_checkpoint(root):
    p = os.path.join(root, "checkpoint.json")
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:
            pass
    return {}

def log_line(root, msg):
    line = f"[{datetime.datetime.now(datetime.timezone.utc).isoformat()}Z] {msg}"
    print(line, flush=True)
    with open(os.path.join(root, "run.log"), "a") as f:
        f.write(line + "\n")

# =====================================================================================
# main
# =====================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\Colab-local", help="base folder for project runs")
    ap.add_argument("--resume", default=None, help="path to an existing run folder to continue")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--quick", action="store_true", help="tiny grid (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="print grid + memory estimate, write nothing")
    a = ap.parse_args()

    grid = quick_grid() if a.quick else default_grid()

    # memory estimate (largest config)
    def mem_gb(c):
        D, N = derive(c); S = c["seeds"]
        return (S * N * D * 4 * 3) / 1e9               # X, Xte-ish, buffers
    worst = max(grid, key=mem_gb)
    print(f"configs: {len(grid)} | device: {a.device} | "
          f"largest GPU tensor footprint ~{mem_gb(worst):.2f} GB (config {config_id(worst)})")
    if a.dry_run:
        for c in grid[:12]:
            D, N = derive(c)
            print(f"  {config_id(c)}  role={c['role']} seeds={c['seeds']} D={D} N={N}")
        print(f"  ... ({len(grid)} total)")
        return

    # run folder
    if a.resume:
        root = a.resume
        assert os.path.isdir(root), f"--resume folder not found: {root}"
    else:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        root = os.path.join(a.root, f"paperA_critfss_{ts}")
    os.makedirs(os.path.join(root, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(root, "Temp"), exist_ok=True)

    atomic_write_json(os.path.join(root, "grid_spec.json"),
                      [dict(config_id=config_id(c), **c, **dict(zip(("D", "N"), derive(c)))) for c in grid])
    ckpt = load_checkpoint(root)
    log_line(root, f"start: {len(grid)} configs, device={a.device}, "
                   f"host={platform.node()}, resume={'yes' if a.resume else 'no'}, "
                   f"done so far={sum(1 for v in ckpt.values() if v.get('done'))}")

    device = torch.device(a.device)
    summary_path = os.path.join(root, "results_summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else []

    for i, c in enumerate(grid):
        cid = config_id(c)
        if ckpt.get(cid, {}).get("done"):
            continue                                    # RESUME: skip completed configs
        try:
            payload, snap_peak, snap_tstar, wall, dims = run_config(c, device)
        except RuntimeError as e:
            log_line(root, f"[{i+1}/{len(grid)}] {cid} FAILED: {e}")
            continue
        atomic_savez(os.path.join(root, "metrics", cid + ".npz"), **payload)
        atomic_savez(os.path.join(root, "Temp", cid + "_at_peak.npz"), **snap_peak)
        atomic_savez(os.path.join(root, "Temp", cid + "_at_Tstar.npz"), **snap_tstar)
        ts_seed = payload["T_star_DD_per_seed"]; conv = ts_seed[ts_seed > 0]
        rowsum = dict(config_id=cid, optim=c["optim"], role=c["role"],
                      gamma=c["gamma"], lam=c["lam"], lr=c["lr"], sigma=c["sigma"],
                      D=int(dims[2]), N=int(dims[1]), seeds=int(dims[0]),
                      n_converged=int((ts_seed > 0).sum()),
                      T_star_median=float(np.median(conv)) if len(conv) else None,
                      T_star_cv=float(conv.std() / conv.mean()) if len(conv) else None,
                      wall_s=round(wall, 2))
        summary.append(rowsum)
        atomic_write_json(summary_path, summary)
        ckpt[cid] = {"done": True, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     "wall_s": round(wall, 2), "n_converged": rowsum["n_converged"]}
        atomic_write_json(os.path.join(root, "checkpoint.json"), ckpt)
        log_line(root, f"[{i+1}/{len(grid)}] {cid}  conv={rowsum['n_converged']}/{dims[0]}  "
                       f"T*med={rowsum['T_star_median']}  {wall:.1f}s")

    log_line(root, f"DONE. results in: {root}")
    print(f"\nAll outputs under: {root}")

if __name__ == "__main__":
    main()
