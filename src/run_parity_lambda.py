#!/usr/bin/env python3
# =====================================================================================
# run_parity_lambda.py  —  rescue probe: regularization sweep on sparse parity.
#
# Follow-up to run_parity.py. At a FIXED task (n=30, k=3, m=1000) we sweep the weight decay lambda and the
# learning rate to answer three questions the first parity run left open:
#   (1) Frontier: does norm@grok vs T_grok trace a regular regularization frontier (as on modular addition)?
#   (2) Regularization-immunity: the earlier clamp-ABOVE delay (2.5x) -- does it lie OFF this frontier
#       (weight decay can only LOWER the norm, so a high-norm point is unreachable by any lambda)?
#   (3) Concentration rescue: at fixed task and lambda, is ||W||_c INVARIANT across learning rate (the clean
#       concentration analog that the m-sweep confounded)? If yes, the norm IS a concentrated target at
#       fixed conditions and the m-drift reflects the task changing, not the norm being ill-defined.
#
# All arms are FREE (no clamp; clamp data already exists from run_parity.py). Same MLP/task/training code as
# run_parity.py (no LayerNorm, no embedding; total norm is the functional norm). 5 criteria: resume-safe,
# atomic, timestamped under D:\Colab-local, rich data, Temp/ snapshots. NO moment reset (no intervention).
# =====================================================================================
import argparse, json, math, os, time, hashlib, datetime, platform
import numpy as np
try:
    import torch
    import torch.nn.functional as F
except Exception:
    print("ERROR: PyTorch required."); raise

def default_grid():
    H, k, n, m, seeds = 512, 3, 30, 1000, 16
    base = dict(optim="adamw", k=k, H=H, n=n, m=m, seeds=seeds, m_test=2000, t_int=10**9, max_steps=50_000, arm="free", rho=1.0)
    grid = []
    for lam in [0.3, 0.5, 1.0, 2.0, 3.0, 5.0]:                 # regularization frontier (lr fixed 1e-3)
        grid.append(dict(base, lam=lam, lr=1e-3))
    for lr in [5e-4, 2e-3]:                                    # lr-invariance at lambda=1 (concentration test)
        grid.append(dict(base, lam=1.0, lr=lr))
    return dedupe(grid)

def quick_grid():
    base = dict(optim="adamw", k=2, H=64, n=12, m=200, seeds=3, m_test=400, t_int=10**9, max_steps=1500, arm="free", rho=1.0)
    return dedupe([dict(base, lam=1.0, lr=2e-3), dict(base, lam=3.0, lr=2e-3)])

def dedupe(grid):
    seen, out = set(), []
    for c in grid:
        kk = config_id(c)
        if kk not in seen: seen.add(kk); out.append(c)
    return out

def base_key(c):
    return f"{c['optim']}_n{c['n']}_k{c['k']}_m{c['m']}_H{c['H']}_wd{c['lam']:.2f}_lr{c['lr']:.0e}"
def config_id(c):
    return (base_key(c) + "__free").replace("+", "")

def log_step_grid(max_steps, n=360):
    pts = np.unique(np.round(np.geomspace(1, max_steps, n)).astype(int))
    return np.unique(np.concatenate([[0], np.clip(pts, 0, max_steps)]))

def spec_entropy(M):
    sg = torch.linalg.svdvals(M); p = sg.pow(2); p = p / (p.sum(-1, keepdim=True) + 1e-30)
    return -(p * (p + 1e-30).log()).sum(-1)

def run_config(c, device, wc=None, dtype=torch.float32, acc_mem=0.99, acc_grok=0.90):
    n, k, S, H = c["n"], c["k"], c["seeds"], c["H"]
    m, m_te = c["m"], c["m_test"]; lam, lr = c["lam"], c["lr"]
    arm, rho, t_int = c["arm"], c["rho"], c["t_int"]
    base_seed = int(hashlib.sha1(base_key(c).encode()).hexdigest(), 16) % (2**31)
    g = torch.Generator(device=device).manual_seed(base_seed)

    # relevant bit subset: fixed (first k) — the model still must discover it from data.
    rel = torch.arange(k, device=device)
    def make(num):
        x = (torch.randint(0, 2, (S, num, n), generator=g, device=device, dtype=dtype) * 2 - 1)   # {-1,+1}
        y = (x[:, :, rel].prod(dim=2) > 0).long()                                                   # parity -> {0,1}
        return x, y
    x_tr, y_tr = make(m)        # m training examples per seed
    x_te, y_te = make(m_te)     # fresh held-out examples (true generalization)

    def param(*shape, scale):
        return (scale * torch.randn(S, *shape, generator=g, device=device, dtype=dtype)).requires_grad_(True)
    W1 = param(n, H, scale=1.0 / math.sqrt(n))
    b1 = param(H,    scale=1e-6)
    W2 = param(H, 2, scale=1.0 / math.sqrt(H))
    b2 = param(2,    scale=1e-6)
    params = [W1, b1, W2, b2]; pnames = ["W1", "b1", "W2", "b2"]
    decay = [True, False, True, False]
    wmats = {"W1": W1, "W2": W2}
    b1m, b2m, aeps = 0.9, 0.999, 1e-8
    ms = [torch.zeros_like(t) for t in params]; vs = [torch.zeros_like(t) for t in params]

    def total_norm():
        return torch.sqrt(sum(wmats[kk].pow(2).sum(dim=tuple(range(1, wmats[kk].ndim))) for kk in wmats))
    def gnorm(names):
        return torch.sqrt(sum(wmats[kk].pow(2).sum(dim=tuple(range(1, wmats[kk].ndim))) for kk in names))
    def rescale_to(target):
        s = target / (total_norm() + 1e-30)
        for kk in wmats:
            sh = [S] + [1] * (wmats[kk].ndim - 1); wmats[kk].mul_(s.view(*sh))

    def loss_acc(x, y):
        h = F.gelu(torch.einsum('sni,sih->snh', x, W1) + b1.unsqueeze(1))
        lg = torch.einsum('snh,shc->snc', h, W2) + b2.unsqueeze(1)
        lp = F.log_softmax(lg, -1)
        ce = -lp.gather(-1, y.unsqueeze(-1)).squeeze(-1).mean(1)
        acc = (lg.argmax(-1) == y).float().mean(1)
        return ce, acc

    steps_to_log = set(int(s) for s in log_step_grid(c["max_steps"]))
    wsnap_steps = sorted(set(int(s) for s in [0, t_int, int(np.sqrt(max(t_int, 1) * c["max_steps"])), c["max_steps"]]))
    keys = ['train_loss', 'test_loss', 'train_acc', 'test_acc', 'weight_norm', 'wn_W1', 'wn_W2', 'eff_rank_W1']
    logged, rec = [], {kk: [] for kk in keys}
    W1_snaps, snap_steps = [], []
    nbi = np.full(S, np.nan, np.float32); nai = np.full(S, np.nan, np.float32)
    diverged = False
    t0 = time.time()
    for step in range(c["max_steps"] + 1):
        if step in steps_to_log:
            with torch.no_grad():
                ce_tr, acc_tr = loss_acc(x_tr, y_tr); ce_te, acc_te = loss_acc(x_te, y_te)
                if not torch.isfinite(ce_tr).all(): diverged = True
                seW = spec_entropy(W1)
                vals = [('train_loss', ce_tr), ('test_loss', ce_te), ('train_acc', acc_tr), ('test_acc', acc_te),
                        ('weight_norm', total_norm()), ('wn_W1', gnorm(["W1"])), ('wn_W2', gnorm(["W2"])),
                        ('eff_rank_W1', seW.exp())]
            logged.append(step)
            for kk, v in vals: rec[kk].append(v.cpu().numpy())
            if diverged: break
        if step in wsnap_steps:
            W1_snaps.append(W1.detach().cpu().numpy().astype(np.float32)); snap_steps.append(step)
        if step == c["max_steps"]: break
        ce_tr, _ = loss_acc(x_tr, y_tr); loss = ce_tr.sum()
        if not torch.isfinite(loss): diverged = True; break
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
                rescale_to(rho * wc)
                if step + 1 == t_int: nai = total_norm().cpu().numpy()

    steps = np.asarray(logged, np.int64)
    out = {kk: np.stack(v, 1).astype(np.float32) for kk, v in rec.items()}
    tr_acc, te_acc = out['train_acc'], out['test_acc']
    Tmem = np.full(S, -1, np.int64); Tgrok = np.full(S, -1, np.int64); wn_at_grok = np.full(S, np.nan, np.float32)
    for s in range(S):
        im = np.where(tr_acc[s] >= acc_mem)[0]; ig = np.where(te_acc[s] >= acc_grok)[0]
        if len(im): Tmem[s] = steps[im[0]]
        if len(ig): Tgrok[s] = steps[ig[0]]; wn_at_grok[s] = out['weight_norm'][s, ig[0]]
    payload = dict(steps=steps, n=np.int64(n), k=np.int64(k), m=np.int64(m), H=np.int64(H),
                   lam=np.float32(lam), lr=np.float32(lr), arm=str(arm), rho=np.float32(rho),
                   t_int=np.int64(t_int), wc_used=np.float32(wc if wc is not None else np.nan),
                   max_steps=np.int64(c["max_steps"]), m_test=np.int64(m_te), diverged=np.bool_(diverged),
                   norm_before_int=nbi, norm_after_int=nai, wn_at_grok=wn_at_grok,
                   T_mem_per_seed=Tmem, T_grok_per_seed=Tgrok, **out)
    snap = dict(W1_snaps=np.stack(W1_snaps, 0) if W1_snaps else np.zeros((0,)),
                snap_steps=np.asarray(snap_steps, np.int64), T_grok=Tgrok, arm=str(arm), rho=np.float32(rho))
    return payload, snap, time.time() - t0, (S, m)

def atomic_write_json(path, obj):
    tmp = path + ".tmp"; json.dump(obj, open(tmp, "w"), indent=1); os.replace(tmp, path)
def atomic_savez(path, **a):
    tmp = path + ".tmp.npz"; np.savez_compressed(tmp, **a); os.replace(tmp, path)
def load_json(p, d): return json.load(open(p)) if os.path.exists(p) else d
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
            f"paperA_paritylam_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
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
        payload, snap, dt, (S, m) = run_config(c, device, wc=None)
        atomic_savez(os.path.join(root, "metrics", cid + ".npz"), **payload)
        atomic_savez(os.path.join(root, "Temp", cid + "_wsnap.npz"), **snap)
        tg = payload["T_grok_per_seed"]; conv = tg[tg > 0]
        row = dict(config_id=cid, arm="free", n=c["n"], k=c["k"], m=c["m"], lam=c["lam"], lr=c["lr"], seeds=S,
                   diverged=bool(payload["diverged"]), grok_frac=float((tg > 0).mean()),
                   T_grok_median=(float(np.median(conv)) if len(conv) else None),
                   wn_at_grok_mean=(float(np.nanmean(payload["wn_at_grok"])) if np.isfinite(payload["wn_at_grok"]).any() else None),
                   wn_W1_at_grok_mean=None, seconds=round(dt, 1))
        # also record W1 norm at grok (functional layer) for the concentration analysis
        wg = payload["wn_W1"]; ta = payload["test_acc"]; w1s = []
        for s in range(S):
            gi = np.where(ta[s] >= 0.90)[0]
            if len(gi): w1s.append(float(wg[s][gi[0]]))
        row["wn_W1_at_grok_mean"] = float(np.mean(w1s)) if w1s else None
        summary = [r for r in summary if r.get("config_id") != cid] + [row]
        atomic_write_json(spath, summary)
        ckpt[cid] = dict(done=True, seconds=round(dt, 1)); atomic_write_json(os.path.join(root, "checkpoint.json"), ckpt)
        flag = " DIVERGED" if payload["diverged"] else ""
        log_line(root, f"[{i+1}/{len(grid)}] {cid}  grok={int((tg>0).sum())}/{S}  "
                       f"Tgrok={row['T_grok_median']}  norm@grok={row['wn_at_grok_mean']}  "
                       f"W1@grok={row['wn_W1_at_grok_mean']}{flag}  {dt:.1f}s")
    log_line(root, "ALL DONE")

if __name__ == "__main__":
    main()
