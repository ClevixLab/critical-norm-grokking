#!/usr/bin/env python3
# =====================================================================================
# run_tfgrok.py  —  Grokking on a REAL 1-layer multi-head attention Transformer.
#
# PURPOSE (universality check for Paper A): reproduce, on an attention-based Transformer
# (not the MLP of run_grokfss.py), the three pillars + the critical-norm finding:
#   (1) delta=1/2-vs-not exponent / FSS collapse over (alpha, p),
#   (2) specialization escape (fourier_gini) at the transition,
#   (3) the CRITICAL WEIGHT NORM at grokking: invariant to alpha, set by wd, scaling with p,
#       + its lr/wd dependence (the ab block) -> timescale/threshold separation.
#
# Architecture: tokens [a, b, =] (vocab=p+1), 1 attention layer (n_heads), parameter-free
# LayerNorm, MLP block, residual stream, unembed at the LAST position. No biases (Nanda-style).
# Batched over seeds S on a leading axis; manual AdamW with decoupled weight decay; full batch.
# Query computed only at the last position (correct & 3x cheaper for a 1-layer predict-at-end model).
#
# Metrics logged on a geometric step grid (per seed): train/test loss & acc, weight_norm
# (sqrt sum-of-squares of ALL weight matrices = Omnigrok control variable), spec_entropy_E,
# spec_entropy_W2 (here = MLP W_in spectrum), eff_rank_E, fourier_gini (embedding). Plus
# T_mem/T_grok/delay per seed, and rich at-grok AND at-mem snapshots (Temp/) for later analysis.
#
# IO: D:\Colab-local\paperA_tfgrok_<timestamp>\{metrics,Temp}, results_summary.json, checkpoint.json,
# grid_spec.json, run.log. Atomic writes (tmp + os.replace). Relaunch with
#   python run_tfgrok.py --resume <folder> --device cuda
# continues exactly where it stopped (config_id depends only on hyperparams, NOT max_steps/seeds).
# Analyze with analyze_runs.py --task grok  (same schema as run_grokfss.py).
# =====================================================================================
import argparse, json, math, os, time, hashlib, datetime, platform
import numpy as np
import torch
import torch.nn.functional as F

# -------------------------------------------------------------------------------------
def default_grid():
    optimizers = ["adamw"]
    d_model, n_heads, n_layers, H = 128, 4, 1, 512
    lam_ref, lr_ref = 1.0, 1e-3
    seeds = 12

    grid = []
    # (A) FSS / criticality + critical-norm-vs-alpha:  p x alpha at reference (wd,lr)
    primes = [53, 71, 97]
    alphas = [0.20, 0.25, 0.30, 0.40, 0.55]
    for opt in optimizers:
        for p in primes:
            for al in alphas:
                grid.append(dict(role="fss", optim=opt, p=p, alpha=al, d_model=d_model,
                                 n_heads=n_heads, n_layers=n_layers, H=H,
                                 lam=lam_ref, lr=lr_ref, seeds=seeds, max_steps=40_000))
    # (B) critical-norm vs (wd, lr): the timescale/threshold-separation test, fixed (p, alpha)
    lams = [0.3, 1.0, 3.0]
    lrs  = [3e-4, 1e-3, 3e-3]
    for opt in optimizers:
        for lam in lams:
            for lr in lrs:
                grid.append(dict(role="ab", optim=opt, p=71, alpha=0.40, d_model=d_model,
                                 n_heads=n_heads, n_layers=n_layers, H=H,
                                 lam=lam, lr=lr, seeds=seeds, max_steps=40_000))
    return dedupe(grid)

def quick_grid():
    # tiny smoke test (CPU): 2 configs, 2 seeds, few steps
    g = [dict(role="fss", optim="adamw", p=13, alpha=0.5, d_model=32, n_heads=2, n_layers=1,
              H=64, lam=1.0, lr=1e-3, seeds=2, max_steps=400),
         dict(role="fss", optim="adamw", p=17, alpha=0.4, d_model=32, n_heads=2, n_layers=1,
              H=64, lam=1.0, lr=1e-3, seeds=2, max_steps=400)]
    return dedupe(g)

def dedupe(grid):
    seen, uniq = set(), []
    for c in grid:
        cid = config_id(c)
        if cid not in seen:
            seen.add(cid); uniq.append(c)
    return uniq

def config_id(c):
    s = (f"{c['optim']}_p{c['p']}_a{c['alpha']:.2f}_d{c['d_model']}_h{c['n_heads']}"
         f"_L{c['n_layers']}_H{c['H']}_wd{c['lam']:.0e}_lr{c['lr']:.0e}")
    return s.replace("+", "")

# -------------------------------------------------------------------------------------
# spectral / mechanistic observables (batched over seed axis S)  -- identical defs to run_grokfss
# -------------------------------------------------------------------------------------
def spec_entropy(M):
    sg = torch.linalg.svdvals(M)
    p = sg.pow(2); p = p / (p.sum(-1, keepdim=True) + 1e-30)
    return -(p * (p + 1e-30).log()).sum(-1)

def fourier_gini(E):
    """E: (S, p, d) -> (S,) Gini of token-frequency power (specialization order parameter)."""
    S, p, d = E.shape
    F_ = torch.fft.rfft(E, dim=1)
    power = (F_.abs() ** 2).sum(-1)[:, 1:]               # drop DC
    power = power / (power.sum(-1, keepdim=True) + 1e-30)
    sorted_p, _ = torch.sort(power, dim=-1)
    n = sorted_p.shape[-1]
    idx = torch.arange(1, n + 1, device=E.device, dtype=E.dtype)
    return (2 * (idx * sorted_p).sum(-1)) / (n * sorted_p.sum(-1) + 1e-30) - (n + 1) / n

def log_step_grid(max_steps, n_points=360):
    pts = np.unique(np.round(np.geomspace(1, max_steps, n_points)).astype(int))
    return np.unique(np.concatenate([[0], np.clip(pts, 0, max_steps)]))

# -------------------------------------------------------------------------------------
# train one config: S seeds batched on GPU; 1-layer attention Transformer; manual AdamW
# -------------------------------------------------------------------------------------
def run_config(c, device, dtype=torch.float32, acc_mem=0.99, acc_grok=0.90):
    p, S = c["p"], c["seeds"]
    d, nh, H = c["d_model"], c["n_heads"], c["H"]
    assert c["n_layers"] == 1, "this runner implements the 1-layer (predict-at-last) Transformer"
    assert d % nh == 0, "d_model must be divisible by n_heads"
    dh = d // nh
    V = p + 1                       # vocab = p numbers + '=' token (index p)
    lam, lr, opt = c["lam"], c["lr"], c["optim"]
    cid = config_id(c)
    base_seed = int(hashlib.sha1(cid.encode()).hexdigest(), 16) % (2**31)
    g = torch.Generator(device=device).manual_seed(base_seed)

    # full dataset of all (a,b) pairs; sequence = [a, b, '=']
    a_all = torch.arange(p, device=device).repeat_interleave(p)
    b_all = torch.arange(p, device=device).repeat(p)
    y_all = (a_all + b_all) % p
    eq = torch.full_like(a_all, p)                                   # '=' token id = p
    toks_all = torch.stack([a_all, b_all, eq], dim=1)                # (M, 3)
    M = p * p
    n_tr = max(2, int(round(c["alpha"] * M)))
    n_te = M - n_tr
    if n_te < 1:
        n_tr = M - 1; n_te = 1

    perm = torch.stack([torch.randperm(M, generator=g, device=device) for _ in range(S)])  # (S,M)
    tr_idx = perm[:, :n_tr]; te_idx = perm[:, n_tr:]
    tok_tr = toks_all[tr_idx]; y_tr = y_all[tr_idx]                  # (S,n_tr,3), (S,n_tr)
    tok_te = toks_all[te_idx]; y_te = y_all[te_idx]

    def param(*shape, scale):
        return (scale * torch.randn(S, *shape, generator=g, device=device, dtype=dtype)).requires_grad_(True)
    E   = param(V, d,  scale=1.0 / math.sqrt(d))
    pos = param(3, d,  scale=0.1 / math.sqrt(d))
    Wq  = param(d, d,  scale=1.0 / math.sqrt(d))
    Wk  = param(d, d,  scale=1.0 / math.sqrt(d))
    Wv  = param(d, d,  scale=1.0 / math.sqrt(d))
    Wo  = param(d, d,  scale=1.0 / math.sqrt(d))
    Win = param(d, H,  scale=1.0 / math.sqrt(d))
    Wout= param(H, d,  scale=1.0 / math.sqrt(H))
    U   = param(d, V,  scale=1.0 / math.sqrt(d))
    params   = [E, pos, Wq, Wk, Wv, Wo, Win, Wout, U]
    pnames   = ["E", "pos", "Wq", "Wk", "Wv", "Wo", "Win", "Wout", "U"]
    decay    = [True]*len(params)                                    # all weight matrices decayed
    norm_set = ["E", "pos", "Wq", "Wk", "Wv", "Wo", "Win", "Wout", "U"]
    if opt == "adamw":
        b1m, b2m, aeps = 0.9, 0.999, 1e-8
        ms = [torch.zeros_like(t) for t in params]
        vs = [torch.zeros_like(t) for t in params]

    def ln(x):                                                       # parameter-free LayerNorm over last dim
        return (x - x.mean(-1, keepdim=True)) / (x.var(-1, unbiased=False, keepdim=True) + 1e-5).sqrt()

    def logits_for(tok):
        # tok: (S,B,3) ; gather embeddings per seed
        B = tok.shape[1]
        idx = tok.reshape(S, B*3, 1).expand(S, B*3, d)
        emb = torch.gather(E, 1, idx).reshape(S, B, 3, d)            # (S,B,3,d)
        x = emb + pos.unsqueeze(1)                                   # add positional
        xln = ln(x)
        # attention: query only the LAST position (predict-at-end, 1 layer)
        qd = torch.einsum('sbd,sde->sbe', xln[:, :, -1, :], Wq).reshape(S, B, nh, dh)      # (S,B,nh,dh)
        kd = torch.einsum('sbtd,sde->sbte', xln, Wk).reshape(S, B, 3, nh, dh)              # (S,B,3,nh,dh)
        vd = torch.einsum('sbtd,sde->sbte', xln, Wv).reshape(S, B, 3, nh, dh)
        scores = torch.einsum('sbhe,sbthe->sbht', qd, kd) / math.sqrt(dh)                  # (S,B,nh,3)
        attn = torch.softmax(scores, dim=-1)
        z = torch.einsum('sbht,sbthe->sbhe', attn, vd).reshape(S, B, d)                    # (S,B,d)
        attn_out = torch.einsum('sbd,sde->sbe', z, Wo)
        x_last = x[:, :, -1, :] + attn_out                                                 # residual
        h = F.gelu(torch.einsum('sbd,sdm->sbm', ln(x_last), Win))
        x_last = x_last + torch.einsum('sbm,smd->sbd', h, Wout)                             # MLP residual
        return torch.einsum('sbd,sdv->sbv', ln(x_last), U)                                 # (S,B,V)

    def loss_acc(tok, y):
        lg = logits_for(tok)
        logp = F.log_softmax(lg, dim=-1)
        ce = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1).mean(dim=1)                      # (S,)
        acc = (lg.argmax(-1) == y).float().mean(dim=1)                                      # (S,)
        return ce, acc

    groups = {"embed": ["E", "pos"], "attn": ["Wq", "Wk", "Wv", "Wo"],
              "mlp": ["Win", "Wout"], "unembed": ["U"]}
    def group_norm(names):
        return torch.sqrt(sum(t.pow(2).sum(dim=tuple(range(1, t.ndim)))
                              for t, nm in zip(params, pnames) if nm in names))              # (S,)
    def weight_norm():
        return group_norm(norm_set)                                                          # (S,) total

    steps_to_log = set(int(s) for s in log_step_grid(c["max_steps"]))
    keys = ['train_loss','test_loss','train_acc','test_acc','weight_norm',
            'wn_embed','wn_attn','wn_mlp','wn_unembed',
            'spec_entropy_E','spec_entropy_W2','eff_rank_E','fourier_gini']
    logged, rec = [], {k: [] for k in keys}
    t0 = time.time()
    for step in range(c["max_steps"] + 1):
        if step in steps_to_log:
            with torch.no_grad():
                ce_tr, acc_tr = loss_acc(tok_tr, y_tr)
                ce_te, acc_te = loss_acc(tok_te, y_te)
                wn  = weight_norm()
                wn_e = group_norm(groups["embed"]); wn_a = group_norm(groups["attn"])
                wn_m = group_norm(groups["mlp"]);   wn_u = group_norm(groups["unembed"])
                seE = spec_entropy(E[:, :p, :])               # spectrum of number-token embeddings
                seW = spec_entropy(Win)                       # MLP input matrix spectrum (W2 slot)
                fg  = fourier_gini(E[:, :p, :])               # specialization order parameter
            logged.append(step)
            for k, val in [('train_loss',ce_tr),('test_loss',ce_te),('train_acc',acc_tr),
                           ('test_acc',acc_te),('weight_norm',wn),
                           ('wn_embed',wn_e),('wn_attn',wn_a),('wn_mlp',wn_m),('wn_unembed',wn_u),
                           ('spec_entropy_E',seE),('spec_entropy_W2',seW),
                           ('eff_rank_E',seE.exp()),('fourier_gini',fg)]:
                rec[k].append(val.detach().cpu().numpy())
        if step == c["max_steps"]:
            break
        ce_tr, _ = loss_acc(tok_tr, y_tr)
        loss = ce_tr.sum()
        for t in params:
            t.grad = None
        loss.backward()
        with torch.no_grad():
            if opt == "adamw":
                t_ = step + 1
                for i, t in enumerate(params):
                    if decay[i]:
                        t.mul_(1 - lr * lam)
                    ms[i].mul_(b1m).add_(t.grad, alpha=1 - b1m)
                    vs[i].mul_(b2m).addcmul_(t.grad, t.grad, value=1 - b2m)
                    mhat = ms[i] / (1 - b1m ** t_); vhat = vs[i] / (1 - b2m ** t_)
                    t.addcdiv_(mhat, vhat.sqrt().add_(aeps), value=-lr)
            else:
                for i, t in enumerate(params):
                    gr = t.grad + (lam * t if decay[i] else 0.0)
                    t.add_(gr, alpha=-lr)

    steps = np.asarray(logged, np.int64)
    out = {k: np.stack(v, axis=1).astype(np.float32) for k, v in rec.items()}              # (S,T)
    tr_acc, te_acc = out['train_acc'], out['test_acc']
    Tmem = np.full(S, -1, np.int64); Tgrok = np.full(S, -1, np.int64)
    for s in range(S):
        im = np.where(tr_acc[s] >= acc_mem)[0]
        if len(im): Tmem[s] = steps[im[0]]
        ig = np.where(te_acc[s] >= acc_grok)[0]
        if len(ig): Tgrok[s] = steps[ig[0]]
    delay = np.where((Tgrok > 0) & (Tmem >= 0), Tgrok - Tmem, -1).astype(np.int64)

    payload = dict(steps=steps, p=np.int64(p), alpha=np.float32(c["alpha"]),
                   d_model=np.int64(d), n_heads=np.int64(nh), n_layers=np.int64(c["n_layers"]),
                   H=np.int64(H), lam=np.float32(lam), lr=np.float32(lr),
                   optim=str(opt), role=str(c["role"]), max_steps=np.int64(c["max_steps"]),
                   n_train=np.int64(n_tr), n_test=np.int64(n_te),
                   acc_mem=np.float32(acc_mem), acc_grok=np.float32(acc_grok),
                   T_mem_per_seed=Tmem, T_grok_per_seed=Tgrok, grok_delay_per_seed=delay, **out)

    # rich snapshots: observables at grok AND at memorization (kept in Temp/ for deep analysis)
    def snap_at(Tev):
        sn = {}
        for s in range(S):
            j = int(np.argmin(np.abs(steps - Tev[s]))) if Tev[s] > 0 else -1
            for k in out:
                sn.setdefault(k, []).append(out[k][s, j] if j >= 0 else np.nan)
        return {k: np.asarray(v, np.float32) for k, v in sn.items()}
    snap = dict(grok=snap_at(Tgrok), mem=snap_at(Tmem),
                T_grok=Tgrok, T_mem=Tmem, p=np.int64(p), alpha=np.float32(c["alpha"]),
                lam=np.float32(lam), lr=np.float32(lr))
    return payload, snap, time.time() - t0, (S, n_tr, p)

# =====================================================================================
# atomic IO / checkpoint  (identical scheme to run_grokfss.py / run_critfss.py)
# =====================================================================================
def atomic_write_json(path, obj):
    tmp = path + ".tmp"; json.dump(obj, open(tmp, "w"), indent=1); os.replace(tmp, path)

def atomic_savez(path, **arrays):
    tmp = path + ".tmp.npz"; np.savez_compressed(tmp, **arrays); os.replace(tmp, path)

def load_checkpoint(root):
    fp = os.path.join(root, "checkpoint.json")
    return json.load(open(fp)) if os.path.exists(fp) else {}

def log_line(root, msg):
    line = f"[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] {msg}"
    print(line, flush=True)
    with open(os.path.join(root, "run.log"), "a") as fh:
        fh.write(line + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=r"D:\Colab-local")
    ap.add_argument("--resume", default=None, help="existing run folder to continue")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quick", action="store_true", help="tiny CPU smoke test")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    grid = quick_grid() if args.quick else default_grid()
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    if args.resume:
        root = args.resume
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        root = os.path.join(args.out_root, f"paperA_tfgrok_{ts}")
    os.makedirs(os.path.join(root, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(root, "Temp"), exist_ok=True)

    print(f"configs: {len(grid)} | device: {device}")
    if args.dry_run:
        for c in grid:
            print("  ", config_id(c), "role=", c["role"], "p=", c["p"], "alpha=", c["alpha"],
                  "wd=", c["lam"], "lr=", c["lr"], "steps=", c["max_steps"])
        return

    atomic_write_json(os.path.join(root, "grid_spec.json"),
                      [dict(config_id=config_id(c), **c) for c in grid])
    ckpt = load_checkpoint(root)
    done = sum(1 for v in ckpt.values() if v.get("done"))
    log_line(root, f"start: {len(grid)} configs, device={device}, host={platform.node()}, "
                   f"resume={'yes' if args.resume else 'no'}, done={done}")

    spath = os.path.join(root, "results_summary.json")
    summary = json.load(open(spath)) if os.path.exists(spath) else []
    for i, c in enumerate(grid):
        cid = config_id(c)
        if ckpt.get(cid, {}).get("done"):
            continue
        payload, snap, dt, (S, n_tr, p) = run_config(c, device)
        atomic_savez(os.path.join(root, "metrics", cid + ".npz"), **payload)
        # snapshots are nested dicts -> save flattened with prefixes (kept in Temp for deep analysis)
        flat = {}
        for grp in ("grok", "mem"):
            for k, v in snap[grp].items():
                flat[f"{grp}__{k}"] = v
        for k in ("T_grok", "T_mem", "p", "alpha", "lam", "lr"):
            flat[k] = snap[k]
        atomic_savez(os.path.join(root, "Temp", cid + "_snapshots.npz"), **flat)

        tg = payload["T_grok_per_seed"]; conv = tg[tg > 0]
        rowsum = dict(config_id=cid, optim=c["optim"], role=c["role"], p=c["p"], alpha=c["alpha"],
                      d_model=c["d_model"], n_heads=c["n_heads"], H=c["H"], lam=c["lam"], lr=c["lr"],
                      seeds=S, n_train=int(n_tr), max_steps=c["max_steps"],
                      grok_frac=float((tg > 0).mean()),
                      T_grok_median=(float(np.median(conv)) if len(conv) else None),
                      norm_at_grok_mean=(float(np.nanmean(snap["grok"]["weight_norm"]))
                                         if np.isfinite(snap["grok"]["weight_norm"]).any() else None),
                      norm_at_grok_std=(float(np.nanstd(snap["grok"]["weight_norm"]))
                                        if np.isfinite(snap["grok"]["weight_norm"]).any() else None),
                      seconds=round(dt, 1))
        summary = [r for r in summary if r.get("config_id") != cid] + [rowsum]
        atomic_write_json(spath, summary)
        ckpt[cid] = dict(done=True, seconds=round(dt, 1))
        atomic_write_json(os.path.join(root, "checkpoint.json"), ckpt)
        log_line(root, f"[{i+1}/{len(grid)}] {cid}  grok={int((tg>0).sum())}/{S}  "
                       f"Tgrok={rowsum['T_grok_median']}  "
                       f"norm@grok={rowsum['norm_at_grok_mean']}  {dt:.1f}s")
    log_line(root, "ALL DONE")

if __name__ == "__main__":
    main()
