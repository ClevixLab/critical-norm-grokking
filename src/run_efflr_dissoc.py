#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
run_efflr_dissoc.py  --  THE arbitration experiment for Paper A
("Which quantity is causal for grokking: the weight-norm STATE, or the effective learning rate?").

THE PROBLEM THIS SOLVES
=======================
Holding the weight norm via a clamp simultaneously changes the *effective learning rate*
(the relative per-step update, ~ lr / ||W|| for AdamW). So every previous norm-clamp result
is CONFOUNDED: "hold norm above wc -> prevents grokking" could be the norm STATE, or could be
the lowered effective LR. The strongest critic (nonstationarity, 2507.20057) says it is the
effective LR, not the norm. This runner breaks the confound with a matched 2-D design.

DESIGN  (clamp matrix: rho x lr)  -- best-practice matched counterfactual on the key confounder
  * Continuous clamp holds ||W|| = rho * wc every step (norm STATE axis = rho).
  * We independently sweep the learning rate lr (effective-LR axis).
  * Measured effective LR is LOGGED directly: eff_lr_X = ||delta_X|| / ||X||  (relative update
    of weight matrix X per step), so we can PROVE which cells are effective-LR-matched and that
    raising lr at a clamp restores/exceeds the baseline effective LR.
  * MATCHED COUNTERFACTUAL: the per-seed init AND data split derive from base_key = (p,alpha,d,H,lam)
    ONLY -- NOT lr, NOT arm, NOT rho -- so every cell of the matrix starts from the identical
    network and sees the identical data. Only (rho, lr) differ. (config_id, for storage/resume,
    includes arm/rho/lr; the SEED does not.)

THE TWO DECISIVE READS  (computed offline by analyze_efflr_dissoc.py)
  (1) ISO-NORM, vary lr (a column of the matrix, esp. the PREVENTED clamp rho=1.30):
      raising lr boosts the effective LR several-fold. If grokking is STILL PREVENTED ->
      prevention is a property of the norm STATE, not the effective LR.  (Kills the critic.)
  (2) ISO-EFFECTIVE-LR, vary rho (cells with matched measured eff_lr but different rho):
      if the OUTCOME (grok / no-grok, and T_grok) differs at matched effective LR ->
      the norm state is the operative variable.
  Ideal result: the prevention boundary in the (rho,lr) plane is ~VERTICAL (set by rho/norm),
  not diagonal (set by eff LR); within the grokking region T_grok falls with lr (eff LR sets
  the RATE). => "norm sets the gate/threshold; effective LR sets the rate." A clean reconciliation.

HONEST / FALSIFIABLE: this experiment CAN refute our thesis. If raising lr rescues grokking at
  the prevented clamp (boundary diagonal), the effective-LR account wins and we must soften the
  norm-causal claim. We report whichever way it falls.

NO ADAMW MOMENT RESET: the continuous clamp re-projects the norm AFTER each optimiser step and
  never zeroes (m,v) -- a reset makes vhat~0 and explodes the first post-reset step, re-growing
  the norm and washing out the very intervention we are testing.

Model: 2-layer MLP on (a+b) mod p, batched over seeds, manual AdamW (identical to run_grokfss/normint).

USAGE (PowerShell):
  python run_efflr_dissoc.py --dry-run
  python run_efflr_dissoc.py --device cuda --out_root D:\Colab-local
  python run_efflr_dissoc.py --resume D:\Colab-local\paperA_efflr_2026...
  python run_efflr_dissoc.py --quick          # tiny CPU smoke test
"""
import argparse, json, math, os, time, hashlib, datetime, platform
import numpy as np
try:
    import torch
    import torch.nn.functional as F
except Exception:
    print("ERROR: PyTorch required."); raise

# critical total norm by p (from the observational run; ||W|| over [E,W1,W2]).
WC = {29: 42.1, 41: 47.6, 59: 54.6, 79: 61.2, 97: 67.1}

def default_grid():
    d, H = 128, 256
    p, alpha, lam, seeds = 59, 0.40, 1.0, 16
    t_int, max_steps = 500, 30_000          # headroom so an lr-rescue (if any) can complete
    wc = WC[p]
    base = dict(optim="adamw", p=p, alpha=alpha, d_model=d, H=H, lam=lam,
                seeds=seeds, t_int=t_int, max_steps=max_steps, wc=wc)
    rhos = [0.85, 1.00, 1.15, 1.30]          # norm-STATE axis (1.30 is prevented at base lr)
    lrs  = [1e-3, 2e-3, 4e-3, 8e-3]          # effective-LR axis (up to 8x base)
    grid = []
    # clamp matrix (the arbitration grid)
    for rho in rhos:
        for lr in lrs:
            grid.append(dict(base, arm="clamp", rho=rho, lr=lr))
    # controls (no clamp) at each lr -> baseline grok time & natural wc crossing per lr
    for lr in lrs:
        grid.append(dict(base, arm="none", rho=1.0, lr=lr))
    return dedupe(grid)

def quick_grid():
    d, H = 32, 64
    base = dict(optim="adamw", p=23, alpha=0.40, d_model=d, H=H, lam=1.0,
                seeds=4, t_int=200, max_steps=4000, wc=30.0)
    return dedupe([dict(base, arm="clamp", rho=1.30, lr=1e-3),
                   dict(base, arm="clamp", rho=1.30, lr=8e-3),   # rescue test cell
                   dict(base, arm="clamp", rho=0.85, lr=1e-3),
                   dict(base, arm="none",  rho=1.0,  lr=1e-3)])

def base_key(c):
    # matched init + data across the WHOLE matrix: depends on task/model/decay ONLY (not lr/arm/rho)
    return f"{c['p']}_{c['alpha']:.2f}_{c['d_model']}_{c['H']}_{c['lam']:.0e}".replace("+", "")

def config_id(c):
    s = (f"{c['optim']}_p{c['p']}_a{c['alpha']:.2f}_d{c['d_model']}_H{c['H']}"
         f"_wd{c['lam']:.0e}_{c['arm']}_rho{c['rho']:.2f}_lr{c['lr']:.0e}")
    return s.replace("+", "")

def dedupe(grid):
    seen, uniq = set(), []
    for c in grid:
        cid = config_id(c)
        if cid not in seen: seen.add(cid); uniq.append(c)
    return uniq

# ----- observables (batched over seed axis S) -----
def spec_entropy(M):
    sg = torch.linalg.svdvals(M); pw = sg.pow(2); pw = pw / (pw.sum(-1, keepdim=True) + 1e-30)
    return -(pw * (pw + 1e-30).log()).sum(-1)

def fourier_gini(E):
    S, p, d = E.shape
    F_ = torch.fft.rfft(E, dim=1); power = (F_.abs() ** 2).sum(-1)[:, 1:]
    power = power / (power.sum(-1, keepdim=True) + 1e-30)
    sp, _ = torch.sort(power, dim=-1); n = sp.shape[-1]
    idx = torch.arange(1, n + 1, device=E.device, dtype=E.dtype)
    return (2 * (idx * sp).sum(-1)) / (n * sp.sum(-1) + 1e-30) - (n + 1) / n

def log_step_grid(max_steps, n_points=360):
    pts = np.unique(np.round(np.geomspace(1, max_steps, n_points)).astype(int))
    return np.unique(np.concatenate([[0], np.clip(pts, 0, max_steps)]))

def frob(t):
    return torch.sqrt(t.pow(2).sum(dim=tuple(range(1, t.ndim))))

# -------------------------------------------------------------------------------------
def run_config(c, device, dtype=torch.float32, acc_mem=0.99, acc_grok=0.90):
    p, S, d, H = c["p"], c["seeds"], c["d_model"], c["H"]
    lam, lr = c["lam"], c["lr"]
    wc, rho, arm, t_int = c["wc"], c["rho"], c["arm"], c["t_int"]
    base_seed = int(hashlib.sha1(base_key(c).encode()).hexdigest(), 16) % (2**31)
    g = torch.Generator(device=device).manual_seed(base_seed)

    a_all = torch.arange(p, device=device).repeat_interleave(p)
    b_all = torch.arange(p, device=device).repeat(p)
    y_all = (a_all + b_all) % p
    M = p * p
    n_tr = max(2, int(round(c["alpha"] * M))); n_te = max(1, M - n_tr)
    perm = torch.stack([torch.randperm(M, generator=g, device=device) for _ in range(S)])
    tr, te = perm[:, :n_tr], perm[:, n_tr:n_tr + n_te]
    a_tr, b_tr, y_tr = a_all[tr], b_all[tr], y_all[tr]
    a_te, b_te, y_te = a_all[te], b_all[te], y_all[te]

    def param(*shape, scale):
        return (scale * torch.randn(S, *shape, generator=g, device=device, dtype=dtype)).requires_grad_(True)
    E  = param(p, d,    scale=1.0 / math.sqrt(d))
    W1 = param(2 * d, H, scale=1.0 / math.sqrt(2 * d))
    b1 = param(H,       scale=1e-6)
    W2 = param(H, p,    scale=1.0 / math.sqrt(H))
    b2 = param(p,       scale=1e-6)
    params = [E, W1, b1, W2, b2]; decay = [True, True, False, True, False]
    wmats = {"E": E, "W1": W1, "W2": W2}
    midx  = {"E": 0, "W1": 1, "W2": 3}                    # index into params for each weight matrix
    b1m, b2m, aeps = 0.9, 0.999, 1e-8
    ms = [torch.zeros_like(t) for t in params]
    vs = [torch.zeros_like(t) for t in params]
    eff = {nm: torch.full((S,), float("nan"), device=device, dtype=dtype)
           for nm in ("E", "W1", "W2")}                  # last measured effective LR per matrix (GPU)

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
    def apply_clamp(target):
        scale = (target / (total_norm() + 1e-30))
        for k in wmats:
            sh = [S] + [1] * (wmats[k].ndim - 1); wmats[k].mul_(scale.view(*sh))

    steps_to_log = set(int(s) for s in log_step_grid(c["max_steps"]))
    ckpt_steps = sorted(set(int(s) for s in np.unique(np.round(
                    np.geomspace(1, c["max_steps"], 8)).astype(int))) | {0, t_int})
    keys = ['train_loss','test_loss','train_acc','test_acc','weight_norm',
            'wn_E','wn_W1','wn_W2','eff_lr_E','eff_lr_W1','eff_lr_W2',
            'spec_entropy_E','eff_rank_E','fourier_gini']
    logged, rec = [], {k: [] for k in keys}
    E_ckpts, E_ckpt_steps = [], []
    norm_at_int = dict(before=None, after=None)
    t0 = time.time()
    for step in range(c["max_steps"] + 1):
        if step in steps_to_log:
            with torch.no_grad():
                ce_tr, acc_tr = loss_acc(a_tr, b_tr, y_tr)
                ce_te, acc_te = loss_acc(a_te, b_te, y_te)
                wn = total_norm(); seE = spec_entropy(E); fg = fourier_gini(E)
                vals = [('train_loss',ce_tr),('test_loss',ce_te),('train_acc',acc_tr),
                        ('test_acc',acc_te),('weight_norm',wn),
                        ('wn_E',gnorm("E")),('wn_W1',gnorm("W1")),('wn_W2',gnorm("W2")),
                        ('eff_lr_E',eff["E"]),('eff_lr_W1',eff["W1"]),('eff_lr_W2',eff["W2"]),
                        ('spec_entropy_E',seE),('eff_rank_E',seE.exp()),('fourier_gini',fg)]
            logged.append(step)
            for k, v in vals: rec[k].append(v.cpu().numpy())
        if step in ckpt_steps:
            E_ckpts.append(E.detach().cpu().numpy().astype(np.float32)); E_ckpt_steps.append(step)
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
                upd = mhat / (vhat.sqrt().add_(aeps))            # AdamW direction*magnitude
                # measure effective LR (relative parameter change) for weight matrices
                for nm, mi in midx.items():
                    if mi == i:
                        eff[nm] = lr * frob(upd) / (frob(t) + 1e-30)
                t.add_(upd, alpha=-lr)
            # continuous clamp AFTER the optimiser step; never reset (m,v)
            if arm == "clamp" and step + 1 >= t_int:
                if norm_at_int["before"] is None:
                    norm_at_int["before"] = total_norm().cpu().numpy()
                apply_clamp(rho * wc)
                if norm_at_int["after"] is None:
                    norm_at_int["after"] = total_norm().cpu().numpy()

    steps = np.asarray(logged, np.int64)
    out = {k: np.stack(v, 1).astype(np.float32) for k, v in rec.items()}
    tr_acc, te_acc = out['train_acc'], out['test_acc']
    Tmem = np.full(S, -1, np.int64); Tgrok = np.full(S, -1, np.int64)
    for s in range(S):
        im = np.where(tr_acc[s] >= acc_mem)[0]; ig = np.where(te_acc[s] >= acc_grok)[0]
        if len(im): Tmem[s] = steps[im[0]]
        if len(ig): Tgrok[s] = steps[ig[0]]
    delay = np.where((Tgrok > 0) & (Tmem >= 0), Tgrok - Tmem, -1).astype(np.int64)

    # post-clamp mean effective LR (steady window after t_int, before grok) -> the matched-pair coordinate
    def mean_eff_post(key):
        idx = [j for j, s in enumerate(steps) if s >= t_int]
        if not idx: return np.full(S, np.nan, np.float32)
        seg = out[key][:, idx]
        return np.nanmean(seg, axis=1).astype(np.float32)
    payload = dict(steps=steps, p=np.int64(p), alpha=np.float32(c["alpha"]),
                   d_model=np.int64(d), H=np.int64(H), lam=np.float32(lam), lr=np.float32(lr),
                   arm=str(arm), rho=np.float32(rho), wc=np.float32(wc),
                   t_int=np.int64(t_int), max_steps=np.int64(c["max_steps"]),
                   n_train=np.int64(n_tr), n_test=np.int64(n_te),
                   T_mem_per_seed=Tmem, T_grok_per_seed=Tgrok, grok_delay_per_seed=delay,
                   eff_lr_E_post=mean_eff_post('eff_lr_E'),
                   eff_lr_W1_post=mean_eff_post('eff_lr_W1'),
                   eff_lr_W2_post=mean_eff_post('eff_lr_W2'),
                   norm_before_int=(norm_at_int["before"] if norm_at_int["before"] is not None
                                    else np.full(S, np.nan, np.float32)),
                   norm_after_int=(norm_at_int["after"] if norm_at_int["after"] is not None
                                   else np.full(S, np.nan, np.float32)), **out)
    snap = dict(E_ckpts=np.stack(E_ckpts, 0), E_ckpt_steps=np.asarray(E_ckpt_steps, np.int64),
                T_grok=Tgrok, T_mem=Tmem, arm=str(arm), rho=np.float32(rho), lr=np.float32(lr),
                wc=np.float32(wc))
    return payload, snap, time.time() - t0, (S, n_tr)

# ----- atomic IO / checkpoint -----
def atomic_write_json(path, obj):
    tmp = path + ".tmp"; json.dump(obj, open(tmp, "w"), indent=1); os.replace(tmp, path)
def atomic_savez(path, **a):
    tmp = path + ".tmp.npz"; np.savez_compressed(tmp, **a); os.replace(tmp, path)
def load_checkpoint(root):
    fp = os.path.join(root, "checkpoint.json"); return json.load(open(fp)) if os.path.exists(fp) else {}
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
    if args.resume: root = args.resume
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        root = os.path.join(args.out_root, f"paperA_efflr_{ts}")
    os.makedirs(os.path.join(root, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(root, "Temp"), exist_ok=True)
    print(f"configs: {len(grid)} | device: {device}")
    if args.dry_run:
        for c in grid: print("  ", config_id(c), "| seed_key:", base_key(c))
        return
    atomic_write_json(os.path.join(root, "grid_spec.json"),
                      [dict(config_id=config_id(c), **c) for c in grid])
    ckpt = load_checkpoint(root); done = sum(1 for v in ckpt.values() if v.get("done"))
    log_line(root, f"start: {len(grid)} configs, device={device}, host={platform.node()}, "
                   f"resume={'yes' if args.resume else 'no'}, done={done}")
    spath = os.path.join(root, "results_summary.json")
    summary = json.load(open(spath)) if os.path.exists(spath) else []
    for i, c in enumerate(grid):
        cid = config_id(c)
        if ckpt.get(cid, {}).get("done"): continue
        payload, snap, dt, (S, n_tr) = run_config(c, device)
        atomic_savez(os.path.join(root, "metrics", cid + ".npz"), **payload)
        atomic_savez(os.path.join(root, "Temp", cid + "_struct.npz"), **snap)
        tg = payload["T_grok_per_seed"]; conv = tg[tg > 0]
        effE = float(np.nanmean(payload["eff_lr_E_post"]))
        row = dict(config_id=cid, arm=c["arm"], rho=c["rho"], lr=c["lr"], wc=c["wc"],
                   p=c["p"], seeds=S, grok_frac=float((tg > 0).mean()),
                   T_grok_median=(float(np.median(conv)) if len(conv) else None),
                   eff_lr_E_post=round(effE, 5), seconds=round(dt, 1))
        summary = [r for r in summary if r.get("config_id") != cid] + [row]
        atomic_write_json(spath, summary)
        ckpt[cid] = dict(done=True, seconds=round(dt, 1))
        atomic_write_json(os.path.join(root, "checkpoint.json"), ckpt)
        log_line(root, f"[{i+1}/{len(grid)}] {cid}  grok={int((tg>0).sum())}/{S}  "
                       f"Tg={row['T_grok_median']}  effLR_E={effE:.4f}  {dt:.1f}s")
    log_line(root, "ALL DONE")

if __name__ == "__main__":
    main()
