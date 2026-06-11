#!/usr/bin/env python3
# =====================================================================================
# run_normint_control.py  —  Does the norm CLAMP gate grokking beyond "more regularization helps"?
#
# Reviewer concern (control A): clamping ||W|| below the critical norm accelerates grokking; maybe that is
# just stronger regularization, not a norm-state "gate". We discriminate with a best-practice design:
#
#   * Build a REGULARIZATION FRONTIER by free training at several fixed weight decays lam in {2,3,5,8}
#     (the canonical "more regularization" knob). Each gives a point (||W||@grok, T_grok).
#   * Place the norm CLAMP points (clamp ||W||=N for several N) against that frontier. If a clamp at norm N
#     groks at a DIFFERENT time than free training whose norm settles at N, then the norm *state* gates
#     grokking beyond regularization. If the clamp points lie ON the free-lam frontier, the effect is "just
#     more regularization" -- which we would report honestly.
#   * OMNIGROK baseline: rescale the initialization to norm N and train freely (Liu et al. 2022) -- the
#     literature's norm manipulation, for direct comparison.
#
# Notes / honesty:
#   - Matched-lam uses FIXED calibrated weight decays (NOT adaptive); the resulting free-norm trajectory
#     only approximately matches a given clamp target, so we compare on the measured (norm, T_grok) plane
#     rather than asserting exact trajectory matching. The lam sweep is chosen to SPAN the clamp norm range.
#   - Weight decay can only LOWER the equilibrium norm, so the clamp-ABOVE-wc arm (which prevents grokking)
#     is inherently not reproducible by regularization -- it is immune to this critique by construction.
#   - The clamp is a per-seed SCALAR rescale: it changes magnitude, not direction. We additionally snapshot
#     the full weight vector at checkpoints (Temp/) so a cross-arm cosine can confirm the clamp does not
#     distort weight geometry relative to control.
#
# Matched counterfactual: seed depends only on base_key (p,alpha,d,H,lr) -- NOT on lam/arm/N -- so control,
# matched-lam, clamp, and omnigrok-init share identical init and data per seed; only the regularization
# mechanism differs. Continuous clamp; NO optimizer-moment reset. MLP identical to run_normint/run_grokfss.
# Resume-safe, atomic, timestamped under D:\Colab-local; rich data + Temp/ snapshots.
# =====================================================================================
import argparse, json, math, os, time, hashlib, datetime, platform
import numpy as np
try:
    import torch
    import torch.nn.functional as F
except Exception:
    print("ERROR: PyTorch required."); raise

WC = 54.6  # measured MLP critical norm at p=59, alpha=0.40, lam=1 (from run_grokfss)

def default_grid():
    d, H = 128, 256
    base = dict(optim="adamw", p=59, alpha=0.40, d_model=d, H=H, lr=1e-3,
                seeds=16, t_int=500, max_steps=20_000, wc=WC)
    grid = [dict(base, arm="free", lam=1.0, N=0.0)]                                  # control
    for lam in [2.0, 3.0, 5.0, 8.0]:                                                 # regularization frontier
        grid.append(dict(base, arm="free", lam=lam, N=0.0))
    for rho in [0.90, 0.82, 0.73]:                                                   # clamp below wc
        grid.append(dict(base, arm="clamp", lam=1.0, N=rho * WC))
    grid.append(dict(base, arm="clamp", lam=1.0, N=1.30 * WC))                       # clamp above (prevention)
    for rho in [0.90, 0.73]:                                                         # Omnigrok rescaled init
        grid.append(dict(base, arm="initrescale", lam=1.0, N=rho * WC))
    return dedupe(grid)

def quick_grid():
    base = dict(optim="adamw", p=17, alpha=0.5, d_model=32, H=64, lr=2e-3,
                seeds=3, t_int=60, max_steps=1200, wc=18.0)
    return dedupe([dict(base, arm="free", lam=1.0, N=0.0),
                   dict(base, arm="free", lam=4.0, N=0.0),
                   dict(base, arm="clamp", lam=1.0, N=0.85 * 18.0),
                   dict(base, arm="initrescale", lam=1.0, N=0.85 * 18.0)])

def dedupe(grid):
    seen, out = set(), []
    for c in grid:
        k = config_id(c)
        if k not in seen: seen.add(k); out.append(c)
    return out

def base_key(c):  # init/data depend ONLY on this (shared by all arms)
    return f"{c['optim']}_p{c['p']}_a{c['alpha']:.2f}_d{c['d_model']}_H{c['H']}_lr{c['lr']:.0e}"

def config_id(c):
    if c["arm"] == "free":          tag = f"free_lam{c['lam']:.1f}"
    elif c["arm"] == "clamp":       tag = f"clamp_N{c['N']:.1f}"
    elif c["arm"] == "initrescale": tag = f"omnigrok_N{c['N']:.1f}"
    else:                           tag = c["arm"]
    return (base_key(c) + "__" + tag).replace("+", "")

def log_step_grid(max_steps, n=360):
    pts = np.unique(np.round(np.geomspace(1, max_steps, n)).astype(int))
    return np.unique(np.concatenate([[0], np.clip(pts, 0, max_steps)]))

def spec_entropy(M):
    sg = torch.linalg.svdvals(M); p = sg.pow(2); p = p / (p.sum(-1, keepdim=True) + 1e-30)
    return -(p * (p + 1e-30).log()).sum(-1)
def fourier_gini(E):
    S, p, d = E.shape
    Fc = torch.fft.rfft(E, dim=1); pw = (Fc.abs() ** 2).sum(-1)[:, 1:]
    pw = pw / (pw.sum(-1, keepdim=True) + 1e-30)
    sp, _ = torch.sort(pw, dim=-1); n = sp.shape[-1]
    idx = torch.arange(1, n + 1, device=E.device, dtype=E.dtype)
    return (2 * (idx * sp).sum(-1)) / (n * sp.sum(-1) + 1e-30) - (n + 1) / n

def run_config(c, device, dtype=torch.float32, acc_mem=0.99, acc_grok=0.90):
    p, S, d, H = c["p"], c["seeds"], c["d_model"], c["H"]
    lam, lr, arm, N, t_int = c["lam"], c["lr"], c["arm"], c["N"], c["t_int"]
    base_seed = int(hashlib.sha1(base_key(c).encode()).hexdigest(), 16) % (2**31)
    g = torch.Generator(device=device).manual_seed(base_seed)

    a_all = torch.arange(p, device=device).repeat_interleave(p)
    b_all = torch.arange(p, device=device).repeat(p)
    y_all = (a_all + b_all) % p
    M = p * p; n_tr = max(2, int(round(c["alpha"] * M))); n_te = M - n_tr
    if n_te < 1: n_tr = M - 1; n_te = 1
    perm = torch.stack([torch.randperm(M, generator=g, device=device) for _ in range(S)])
    tr, te = perm[:, :n_tr], perm[:, n_tr:]
    a_tr, b_tr, y_tr = a_all[tr], b_all[tr], y_all[tr]
    a_te, b_te, y_te = a_all[te], b_all[te], y_all[te]

    def param(*shape, scale):
        return (scale * torch.randn(S, *shape, generator=g, device=device, dtype=dtype)).requires_grad_(True)
    E  = param(p, d,    scale=1.0 / math.sqrt(d))
    W1 = param(2 * d, H, scale=1.0 / math.sqrt(2 * d))
    b1 = param(H,       scale=1e-6)
    W2 = param(H, p,    scale=1.0 / math.sqrt(H))
    b2 = param(p,       scale=1e-6)
    params = [E, W1, b1, W2, b2]; pnames = ["E", "W1", "b1", "W2", "b2"]
    decay = [True, True, False, True, False]
    wmats = {"E": E, "W1": W1, "W2": W2}
    b1m, b2m, aeps = 0.9, 0.999, 1e-8
    ms = [torch.zeros_like(t) for t in params]; vs = [torch.zeros_like(t) for t in params]

    def total_norm():
        return torch.sqrt(sum(wmats[k].pow(2).sum(dim=tuple(range(1, wmats[k].ndim))) for k in wmats))
    def gnorm(names):
        return torch.sqrt(sum(wmats[k].pow(2).sum(dim=tuple(range(1, wmats[k].ndim))) for k in names))
    def rescale_to(target):                       # per-seed scalar rescale -> ||W|| = target (preserves direction)
        s = target / (total_norm() + 1e-30)
        for k in wmats:
            sh = [S] + [1] * (wmats[k].ndim - 1); wmats[k].mul_(s.view(*sh))
    def wvec():                                   # flattened weight vector per seed (E,W1,W2) -> (S, P)
        return torch.cat([wmats[k].reshape(S, -1) for k in ["E", "W1", "W2"]], dim=1)

    if arm == "initrescale":                      # Omnigrok-style: start at norm N, train freely
        with torch.no_grad(): rescale_to(N)

    def forward(ai, bi):
        ea = torch.gather(E, 1, ai.unsqueeze(-1).expand(S, ai.shape[1], d))
        eb = torch.gather(E, 1, bi.unsqueeze(-1).expand(S, bi.shape[1], d))
        x = torch.cat([ea, eb], -1)
        h = F.gelu(torch.einsum('sbe,seh->sbh', x, W1) + b1.unsqueeze(1))
        return torch.einsum('sbh,shp->sbp', h, W2) + b2.unsqueeze(1)
    def loss_acc(ai, bi, y):
        lg = forward(ai, bi); lp = F.log_softmax(lg, -1)
        ce = -lp.gather(-1, y.unsqueeze(-1)).squeeze(-1).mean(1)
        acc = (lg.argmax(-1) == y).float().mean(1)
        return ce, acc

    steps_to_log = set(int(s) for s in log_step_grid(c["max_steps"]))
    wsnap_steps = sorted(set(int(s) for s in [0, t_int,
                         int(np.sqrt(t_int * c["max_steps"])), c["max_steps"]]))
    keys = ['train_loss','test_loss','train_acc','test_acc','weight_norm',
            'wn_E','wn_W1','wn_W2','fourier_gini','eff_rank_E']
    logged, rec = [], {k: [] for k in keys}
    W_snaps, W_snap_steps = [], []
    t0 = time.time()
    for step in range(c["max_steps"] + 1):
        if step in steps_to_log:
            with torch.no_grad():
                ce_tr, acc_tr = loss_acc(a_tr, b_tr, y_tr)
                ce_te, acc_te = loss_acc(a_te, b_te, y_te)
                seE = spec_entropy(E); fg = fourier_gini(E)
                vals = [('train_loss',ce_tr),('test_loss',ce_te),('train_acc',acc_tr),('test_acc',acc_te),
                        ('weight_norm',total_norm()),('wn_E',gnorm(["E"])),('wn_W1',gnorm(["W1"])),
                        ('wn_W2',gnorm(["W2"])),('fourier_gini',fg),('eff_rank_E',seE.exp())]
            logged.append(step)
            for k, v in vals: rec[k].append(v.cpu().numpy())
        if step in wsnap_steps:
            W_snaps.append(wvec().detach().cpu().numpy().astype(np.float32)); W_snap_steps.append(step)
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
            if arm == "clamp" and step + 1 >= t_int:
                rescale_to(N)

    steps = np.asarray(logged, np.int64)
    out = {k: np.stack(v, 1).astype(np.float32) for k, v in rec.items()}
    tr_acc, te_acc = out['train_acc'], out['test_acc']
    Tmem = np.full(S, -1, np.int64); Tgrok = np.full(S, -1, np.int64); wn_at_grok = np.full(S, np.nan, np.float32)
    for s in range(S):
        im = np.where(tr_acc[s] >= acc_mem)[0]; ig = np.where(te_acc[s] >= acc_grok)[0]
        if len(im): Tmem[s] = steps[im[0]]
        if len(ig): Tgrok[s] = steps[ig[0]]; wn_at_grok[s] = out['weight_norm'][s, ig[0]]
    payload = dict(steps=steps, p=np.int64(p), alpha=np.float32(c["alpha"]),
                   d_model=np.int64(d), H=np.int64(H), lr=np.float32(lr),
                   arm=str(arm), lam=np.float32(lam), N=np.float32(N), wc=np.float32(c["wc"]),
                   t_int=np.int64(t_int), max_steps=np.int64(c["max_steps"]),
                   n_train=np.int64(n_tr), n_test=np.int64(n_te),
                   T_mem_per_seed=Tmem, T_grok_per_seed=Tgrok, wn_at_grok=wn_at_grok, **out)
    snap = dict(W_snaps=np.stack(W_snaps, 0), W_snap_steps=np.asarray(W_snap_steps, np.int64),
                arm=str(arm), lam=np.float32(lam), N=np.float32(N), T_grok=Tgrok)
    return payload, snap, time.time() - t0, (S, n_tr)

def atomic_write_json(path, obj):
    tmp = path + ".tmp"; json.dump(obj, open(tmp, "w"), indent=1); os.replace(tmp, path)
def atomic_savez(path, **a):
    tmp = path + ".tmp.npz"; np.savez_compressed(tmp, **a); os.replace(tmp, path)
def load_json(p, default): return json.load(open(p)) if os.path.exists(p) else default
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
            f"paperA_normctl_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(os.path.join(root, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(root, "Temp"), exist_ok=True)
    print(f"configs: {len(grid)} | device: {device}")
    if args.dry_run:
        for c in grid: print("  ", config_id(c), "| seed_key:", base_key(c))
        return
    atomic_write_json(os.path.join(root, "grid_spec.json"), [dict(config_id=config_id(c), **c) for c in grid])
    ckpt = load_json(os.path.join(root, "checkpoint.json"), {})
    log_line(root, f"start: {len(grid)} configs, device={device}, host={platform.node()}, "
                   f"resume={'yes' if args.resume else 'no'}, done={sum(1 for v in ckpt.values() if v.get('done'))}")
    spath = os.path.join(root, "results_summary.json"); summary = load_json(spath, [])
    for i, c in enumerate(grid):
        cid = config_id(c)
        if ckpt.get(cid, {}).get("done"): continue
        payload, snap, dt, (S, n_tr) = run_config(c, device)
        atomic_savez(os.path.join(root, "metrics", cid + ".npz"), **payload)
        atomic_savez(os.path.join(root, "Temp", cid + "_wsnap.npz"), **snap)
        tg = payload["T_grok_per_seed"]; conv = tg[tg > 0]
        row = dict(config_id=cid, arm=c["arm"], lam=c["lam"], N=c["N"], p=c["p"], alpha=c["alpha"], seeds=S,
                   grok_frac=float((tg > 0).mean()),
                   T_grok_median=(float(np.median(conv)) if len(conv) else None),
                   wn_at_grok_mean=(float(np.nanmean(payload["wn_at_grok"])) if np.isfinite(payload["wn_at_grok"]).any() else None),
                   seconds=round(dt, 1))
        summary = [r for r in summary if r.get("config_id") != cid] + [row]
        atomic_write_json(spath, summary)
        ckpt[cid] = dict(done=True, seconds=round(dt, 1)); atomic_write_json(os.path.join(root, "checkpoint.json"), ckpt)
        log_line(root, f"[{i+1}/{len(grid)}] {cid}  grok={int((tg>0).sum())}/{S}  "
                       f"Tgrok={row['T_grok_median']}  norm@grok={row['wn_at_grok_mean']}  {dt:.1f}s")
    log_line(root, "ALL DONE")

if __name__ == "__main__":
    main()
