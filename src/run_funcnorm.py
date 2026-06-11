#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
run_funcnorm.py  --  Two targeted experiments that turn the re-analysis findings into
laws, for Paper A ("When is the weight norm causal for grokking?").

Both grids reuse the canonical grokfss MLP/AdamW machinery (same model, same batched-
over-seeds GPU loop, same resumable timestamped storage) and ADD per-layer and
*intensive* (scale-normalised) norm logging that the original grokfss runner did not save.

WHY THESE TWO RUNS
==================
Re-analysis of the existing normint/grokfss data established (verified, this repo):
  * Bidirectional washout: forcing the post-memorisation norm anywhere in [38,71] still
    groks at ||W|| ~ 54-55 (CV ~1.1%); the EMBEDDING norm is the sharpest invariant
    (CV ~0.52% vs 1.15% total) -> the operative quantity looks like the norm of the
    matrix that sets the function's I/O scale (embedding here, unembedding in the LN
    transformer), not the bare total norm.
  * The total critical norm scales as ||W||_c ~ p^0.38 (grokfss), but per-layer / intensive
    scaling was never logged, so we cannot yet say which norm carries the clean law.

Run (a) SCALE: observational sweep over many primes p at the reference config, logging
  per-layer norms (E, W1, W2) and intensive norms (per-token-row RMS of E, per-class-column
  RMS of W2, per-element RMS). GOAL: (i) confirm the embedding norm is the tight invariant
  across p; (ii) find which norm is scale-free (p-invariant) vs which carries the exponent
  -> a *function-scale* critical norm law. No intervention; nothing here can be "forced".

Run (b) OPTWD: 2x2(-ish) optimiser x weight-decay contrast at the reference (p,alpha),
  AdamW/SGD x {weight decay, none}. GOAL: test whether the critical (function-scale) norm
  and grokking itself are specific to AdamW + weight decay, or hold more generally. This
  directly engages the contested literature:
    - Golechha 2024 / Notsawo 2025 / "Data Falls Short" 2025: grokking WITHOUT weight decay
      with INCREASING norms;  - Nonstationarity 2025: effective LR, not norm, drives grokking.
  HONEST DESIGN: SGD / no-WD may NOT grok within budget (censored T_grok = -1 is a valid,
  informative outcome, consistent with a norm-relaxation account); we do not tune toward a
  desired answer. The decisive read-outs are computed offline from the saved curves:
    (1) does grokking occur?  (2) if it groks without weight decay, does the norm rise or
    fall to reach grokking, and does it cross the SAME function-scale critical value (i.e.
    is the critical norm an attractor approached from EITHER direction)?  (3) is the
    embedding/function-scale norm at grok the same across optimiser/WD conditions?

The two grids share one timestamped folder and one config_id space, so a single
`--resume <folder>` continues whichever configs are unfinished.

USAGE (PowerShell):
  python run_funcnorm.py --dry-run
  python run_funcnorm.py --device cuda --root D:\Colab-local
  python run_funcnorm.py --device cuda --root D:\Colab-local --only scale     # run (a) only
  python run_funcnorm.py --device cuda --root D:\Colab-local --only optwd     # run (b) only
  python run_funcnorm.py --resume D:\Colab-local\paperA_funcnorm_2026...      # kill-safe resume
  python run_funcnorm.py --quick                                              # tiny CPU smoke test

NOTES
  * No-AdamW-moment-reset rule is irrelevant here: these are plain training runs, no in-place
    norm intervention. (That rule only applies to the clamp/rescale runners.)
  * Decoupled weight decay for AdamW; coupled L2 for SGD; never on biases. lam=0 => no decay.
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
# grids
# =====================================================================================
def default_grid(which="both"):
    d_model, H = 128, 256
    seeds = 20
    grid = []

    # ---- (a) SCALE: function-scale norm vs p (observational, reference optimiser) ----
    if which in ("both", "scale"):
        primes = [23, 29, 41, 53, 59, 67, 79, 89, 97, 113]   # 10-point lever arm for a clean fit
        for p in primes:
            # large p needs a longer budget to grok at alpha=0.40; keep censoring visible if not
            ms = 60_000 if p <= 79 else 100_000
            grid.append(dict(role="scale", optim="adamw", p=p, alpha=0.40, d_model=d_model, H=H,
                             lam=1.0, lr=1e-3, seeds=seeds, max_steps=ms))

    # ---- (b) OPTWD: optimiser x weight-decay contrast at the reference (p, alpha) ----
    if which in ("both", "optwd"):
        p0, al0 = 59, 0.40
        # AdamW with weight decay (matched baseline; reproduces normint ctrl regime)
        grid.append(dict(role="optwd", optim="adamw", p=p0, alpha=al0, d_model=d_model, H=H,
                         lam=1.0, lr=1e-3, seeds=seeds, max_steps=60_000))
        # AdamW WITHOUT weight decay (does it grok? norm direction?) - long budget
        grid.append(dict(role="optwd", optim="adamw", p=p0, alpha=al0, d_model=d_model, H=H,
                         lam=0.0, lr=1e-3, seeds=seeds, max_steps=200_000))
        # SGD with coupled L2 (two lrs; SGD groks slowly if at all)
        for lr in (1e-2, 3e-2):
            grid.append(dict(role="optwd", optim="sgd", p=p0, alpha=al0, d_model=d_model, H=H,
                             lam=1.0, lr=lr, seeds=seeds, max_steps=200_000))
        # SGD WITHOUT regularisation (two lrs)
        for lr in (1e-2, 3e-2):
            grid.append(dict(role="optwd", optim="sgd", p=p0, alpha=al0, d_model=d_model, H=H,
                             lam=0.0, lr=lr, seeds=seeds, max_steps=200_000))

    # dedupe by config_id
    seen, uniq = set(), []
    for c in grid:
        cid = config_id(c)
        if cid not in seen:
            seen.add(cid); uniq.append(c)
    return uniq

def quick_grid():
    return [dict(role="scale", optim="adamw", p=23, alpha=0.40, d_model=64, H=128,
                 lam=1.0, lr=3e-3, seeds=4, max_steps=6000),
            dict(role="scale", optim="adamw", p=29, alpha=0.40, d_model=64, H=128,
                 lam=1.0, lr=3e-3, seeds=4, max_steps=6000),
            dict(role="optwd", optim="sgd", p=23, alpha=0.40, d_model=64, H=128,
                 lam=0.0, lr=3e-2, seeds=4, max_steps=6000)]

def config_id(c):
    # hyperparams ONLY (NOT seeds / max_steps) so extending budget/seeds stays resume-compatible
    s = (f"{c['optim']}_p{c['p']}_a{c['alpha']:.2f}_d{c['d_model']}_H{c['H']}"
         f"_wd{c['lam']:.0e}_lr{c['lr']:.0e}_{c['role']}")
    return s.replace("+", "")

# =====================================================================================
# spectral / mechanistic observables (batched over seed axis S)
# =====================================================================================
def spec_entropy(M):
    sg = torch.linalg.svdvals(M)
    pw = sg.pow(2); pw = pw / (pw.sum(-1, keepdim=True) + 1e-30)
    return -(pw * (pw + 1e-30).log()).sum(-1)

def fourier_gini(E):
    S, p, d = E.shape
    F_ = torch.fft.rfft(E, dim=1)
    power = (F_.abs() ** 2).sum(-1)[:, 1:]
    power = power / (power.sum(-1, keepdim=True) + 1e-30)
    sorted_p, _ = torch.sort(power, dim=-1)
    n = sorted_p.shape[-1]
    idx = torch.arange(1, n + 1, device=E.device, dtype=E.dtype)
    return (2 * (idx * sorted_p).sum(-1)) / (n * sorted_p.sum(-1) + 1e-30) - (n + 1) / n

def log_step_grid(max_steps, n_points=360):
    pts = np.unique(np.round(np.geomspace(1, max_steps, n_points)).astype(int))
    return np.unique(np.concatenate([[0], np.clip(pts, 0, max_steps)]))

def fro(t):
    """per-seed Frobenius norm over all non-seed axes -> (S,)."""
    return torch.sqrt(t.pow(2).sum(dim=tuple(range(1, t.ndim))))

# =====================================================================================
# train one config (all S seeds batched on GPU via autograd; manual AdamW / SGD)
# =====================================================================================
def run_config(c, device, dtype=torch.float32, acc_mem=0.99, acc_grok=0.90):
    p, S = c["p"], c["seeds"]
    d, H = c["d_model"], c["H"]
    lam, lr, opt = c["lam"], c["lr"], c["optim"]
    cid = config_id(c)
    base_seed = int(hashlib.sha1(cid.encode()).hexdigest(), 16) % (2**31)
    g = torch.Generator(device=device).manual_seed(base_seed)

    a_all = torch.arange(p, device=device).repeat_interleave(p)
    b_all = torch.arange(p, device=device).repeat(p)
    y_all = (a_all + b_all) % p
    M = p * p
    n_tr = max(2, int(round(c["alpha"] * M))); n_te = M - n_tr
    if n_te < 1: n_tr = M - 1; n_te = 1

    perm = torch.stack([torch.randperm(M, generator=g, device=device) for _ in range(S)])
    tr_idx = perm[:, :n_tr]; te_idx = perm[:, n_tr:]
    a_tr = a_all[tr_idx]; b_tr = b_all[tr_idx]; y_tr = y_all[tr_idx]
    a_te = a_all[te_idx]; b_te = b_all[te_idx]; y_te = y_all[te_idx]

    def param(*shape, scale):
        return (scale * torch.randn(S, *shape, generator=g, device=device, dtype=dtype)).requires_grad_(True)
    E  = param(p, d,    scale=1.0 / math.sqrt(d))
    W1 = param(2 * d, H, scale=1.0 / math.sqrt(2 * d))
    b1 = param(H,       scale=1e-6)
    W2 = param(H, p,    scale=1.0 / math.sqrt(H))
    b2 = param(p,       scale=1e-6)
    params = [E, W1, b1, W2, b2]
    decay_mask = [True, True, False, True, False]
    if opt == "adamw":
        b1m, b2m, aeps = 0.9, 0.999, 1e-8
        ms = [torch.zeros_like(t) for t in params]
        vs = [torch.zeros_like(t) for t in params]

    def forward(a_idx, b_idx):
        ea = torch.gather(E, 1, a_idx.unsqueeze(-1).expand(S, a_idx.shape[1], d))
        eb = torch.gather(E, 1, b_idx.unsqueeze(-1).expand(S, b_idx.shape[1], d))
        x = torch.cat([ea, eb], dim=-1)
        h = F.gelu(torch.einsum('sbe,seh->sbh', x, W1) + b1.unsqueeze(1))
        return torch.einsum('sbh,shp->sbp', h, W2) + b2.unsqueeze(1)

    def loss_acc(a_idx, b_idx, y):
        logits = forward(a_idx, b_idx)
        logp = F.log_softmax(logits, dim=-1)
        ce = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1).mean(dim=1)
        acc = (logits.argmax(-1) == y).float().mean(dim=1)
        return ce, acc

    steps_to_log = set(int(s) for s in log_step_grid(c["max_steps"]))
    keys = ['train_loss', 'test_loss', 'train_acc', 'test_acc',
            'weight_norm', 'wn_E', 'wn_W1', 'wn_W2',
            'e_rowrms', 'w2_colrms', 'e_elemrms',          # intensive (scale-normalised) norms
            'spec_entropy_E', 'eff_rank_E', 'fourier_gini']
    logged, rec = [], {k: [] for k in keys}
    t0 = time.time()
    for step in range(c["max_steps"] + 1):
        if step in steps_to_log:
            with torch.no_grad():
                ce_tr, acc_tr = loss_acc(a_tr, b_tr, y_tr)
                ce_te, acc_te = loss_acc(a_te, b_te, y_te)
                nE, nW1, nW2 = fro(E), fro(W1), fro(W2)
                wn = torch.sqrt(nE**2 + nW1**2 + nW2**2)
                # intensive norms: per-token embedding row scale, per-class readout column scale,
                # per-element embedding scale (all scale-free candidates for a p-invariant law)
                e_rowrms = nE / math.sqrt(p)            # ||E||_F / sqrt(#tokens)
                w2_colrms = nW2 / math.sqrt(p)          # ||W2||_F / sqrt(#classes)
                e_elemrms = nE / math.sqrt(p * d)       # ||E||_F / sqrt(#elements)
                seE = spec_entropy(E); fg = fourier_gini(E)
            logged.append(step)
            for k, val in [('train_loss', ce_tr), ('test_loss', ce_te), ('train_acc', acc_tr),
                           ('test_acc', acc_te), ('weight_norm', wn), ('wn_E', nE),
                           ('wn_W1', nW1), ('wn_W2', nW2), ('e_rowrms', e_rowrms),
                           ('w2_colrms', w2_colrms), ('e_elemrms', e_elemrms),
                           ('spec_entropy_E', seE), ('eff_rank_E', seE.exp()), ('fourier_gini', fg)]:
                rec[k].append(val.cpu().numpy())
        if step == c["max_steps"]:
            break
        ce_tr, _ = loss_acc(a_tr, b_tr, y_tr)
        loss = ce_tr.sum()
        for t in params:
            if t.grad is not None: t.grad = None
        loss.backward()
        with torch.no_grad():
            if opt == "adamw":
                t_ = step + 1
                for i, t in enumerate(params):
                    if decay_mask[i] and lam > 0:
                        t.mul_(1 - lr * lam)                       # decoupled weight decay (lam=0 -> none)
                    ms[i].mul_(b1m).add_(t.grad, alpha=1 - b1m)
                    vs[i].mul_(b2m).addcmul_(t.grad, t.grad, value=1 - b2m)
                    mhat = ms[i] / (1 - b1m ** t_); vhat = vs[i] / (1 - b2m ** t_)
                    t.addcdiv_(mhat, vhat.sqrt().add_(aeps), value=-lr)
            else:  # sgd + coupled L2 (lam=0 -> plain SGD)
                for i, t in enumerate(params):
                    gr = t.grad + (lam * t if (decay_mask[i] and lam > 0) else 0.0)
                    t.add_(gr, alpha=-lr)

    steps = np.asarray(logged, np.int64)
    out = {k: np.stack(v, axis=1).astype(np.float32) for k, v in rec.items()}
    tr_acc = out['train_acc']; te_acc = out['test_acc']
    Tmem = np.full(S, -1, np.int64); Tgrok = np.full(S, -1, np.int64)
    for s in range(S):
        im = np.where(tr_acc[s] >= acc_mem)[0]
        if len(im): Tmem[s] = steps[im[0]]
        ig = np.where(te_acc[s] >= acc_grok)[0]
        if len(ig): Tgrok[s] = steps[ig[0]]
    delay = np.where((Tgrok > 0) & (Tmem >= 0), Tgrok - Tmem, -1).astype(np.int64)

    # extra offline-ready summaries: norm-at-grok per layer, and norm DIRECTION
    # (norm at mem vs at grok) -> tells us whether grok is approached from below or above.
    def at(idx_per_seed, key):
        v = []
        for s in range(S):
            j = idx_per_seed[s]
            v.append(out[key][s, j] if j >= 0 else np.nan)
        return np.asarray(v, np.float32)
    jgrok = np.array([int(np.argmin(np.abs(steps - Tgrok[s]))) if Tgrok[s] > 0 else -1 for s in range(S)])
    jmem  = np.array([int(np.argmin(np.abs(steps - Tmem[s])))  if Tmem[s]  > 0 else -1 for s in range(S)])
    norm_dir = {}
    for key in ['weight_norm', 'wn_E', 'wn_W1', 'wn_W2', 'e_rowrms', 'w2_colrms']:
        norm_dir[f'{key}__at_mem']  = at(jmem, key)
        norm_dir[f'{key}__at_grok'] = at(jgrok, key)

    payload = dict(steps=steps, p=np.int64(p), alpha=np.float32(c["alpha"]),
                   d_model=np.int64(d), H=np.int64(H), lam=np.float32(lam), lr=np.float32(lr),
                   optim=str(opt), role=str(c["role"]), max_steps=np.int64(c["max_steps"]),
                   n_train=np.int64(n_tr), n_test=np.int64(n_te),
                   acc_mem=np.float32(acc_mem), acc_grok=np.float32(acc_grok),
                   T_mem_per_seed=Tmem, T_grok_per_seed=Tgrok, grok_delay_per_seed=delay,
                   **norm_dir, **out)
    snap = {}
    for s in range(S):
        j = jgrok[s]
        for k in out:
            snap.setdefault(k, []).append(out[k][s, j] if j >= 0 else np.nan)
    snap = {k: np.asarray(v, np.float32) for k, v in snap.items()}
    return payload, snap, time.time() - t0, (S, n_tr, p)

# =====================================================================================
# atomic IO / checkpoint
# =====================================================================================
def atomic_write_json(path, obj):
    tmp = path + ".tmp"; json.dump(obj, open(tmp, "w"), indent=1); os.replace(tmp, path)

def atomic_savez(path, **arrays):
    tmp = path + ".tmp.npz"; np.savez_compressed(tmp, **arrays); os.replace(tmp, path)

def load_checkpoint(root):
    pth = os.path.join(root, "checkpoint.json")
    try:
        return json.load(open(pth)) if os.path.exists(pth) else {}
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
    ap.add_argument("--only", choices=["both", "scale", "optwd"], default="both")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    grid = quick_grid() if a.quick else default_grid(a.only)
    print(f"configs: {len(grid)} | device: {a.device} | only={a.only}")
    if a.dry_run:
        for c in grid:
            print(f"  {config_id(c):44s} role={c['role']:6s} p={c['p']:3d} opt={c['optim']:5s} "
                  f"wd={c['lam']} lr={c['lr']:.0e} seeds={c['seeds']} steps={c['max_steps']}")
        return

    if a.resume:
        root = a.resume; assert os.path.isdir(root), f"--resume not found: {root}"
    else:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        root = os.path.join(a.root, f"paperA_funcnorm_{ts}")
    os.makedirs(os.path.join(root, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(root, "Temp"), exist_ok=True)
    atomic_write_json(os.path.join(root, "grid_spec.json"),
                      [dict(config_id=config_id(c), **c) for c in grid])
    ckpt = load_checkpoint(root)
    log_line(root, f"start: {len(grid)} configs, device={a.device}, host={platform.node()}, "
                   f"resume={'yes' if a.resume else 'no'}, "
                   f"done={sum(1 for v in ckpt.values() if v.get('done'))}")
    device = torch.device(a.device)
    summary = []
    for ci, c in enumerate(grid):
        cid = config_id(c)
        if ckpt.get(cid, {}).get("done"):
            log_line(root, f"skip {cid} (done)"); continue
        try:
            payload, snap, secs, shape = run_config(c, device)
            atomic_savez(os.path.join(root, "metrics", f"{cid}.npz"), **payload)
            atomic_savez(os.path.join(root, "Temp", f"{cid}_at_grok.npz"), **snap)
            ng = int(((payload['T_grok_per_seed'] > 0)).sum()); S = c['seeds']
            wgrok = payload['weight_norm__at_grok']; egrok = payload['wn_E__at_grok']
            mn = float(np.nanmean(wgrok)) if ng else float('nan')
            men = float(np.nanmean(egrok)) if ng else float('nan')
            ckpt[cid] = dict(done=True, secs=round(secs, 1), grok=f"{ng}/{S}",
                             wn_grok=round(mn, 3), wnE_grok=round(men, 3))
            atomic_write_json(os.path.join(root, "checkpoint.json"), ckpt)
            summary.append(dict(config_id=cid, **ckpt[cid]))
            log_line(root, f"[{ci+1}/{len(grid)}] {cid}  grok {ng}/{S}  "
                           f"||W||@grok={mn:.2f} ||E||@grok={men:.2f}  ({secs:.1f}s, shape={shape})")
        except Exception as e:
            log_line(root, f"ERROR {cid}: {repr(e)}")
            raise
    atomic_write_json(os.path.join(root, "results_summary.json"), summary)
    log_line(root, f"DONE: {len(summary)} configs this session. root={root}")

if __name__ == "__main__":
    main()
