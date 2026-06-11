#!/usr/bin/env python3
# =====================================================================================
# run_normint.py  —  CAUSAL test of the critical-norm hypothesis for grokking.
#
# Hypothesis: grokking occurs when ||W|| relaxes to a critical value ||W||_c.
# Best-practice causal design (Omnigrok-style norm rescaling + matched counterfactuals):
#   * MATCHED COUNTERFACTUAL: base_seed derives from base_key (p,alpha,d,H,lam,lr) ONLY,
#     so the control and every intervention share identical init + data per seed and the
#     same pre-intervention trajectory; only the intervention differs. (config_id, used for
#     storage/resume, includes the arm — but the seed does not.)
#   * Arms:
#       none      : control (no intervention) -> baseline T_grok.
#       rescale   : one-shot at step t_int, project ALL weights to ||W|| = rho * wc
#                   (dose-response over rho), reset AdamW moments. Prediction: smaller rho
#                   (nearer/below wc) -> faster grok; larger rho -> slower.  Monotone = causal.
#       layer     : one-shot at t_int, scale ONLY one group {all,embed,hidden,out} by factor f.
#                   Localizes which norm matters (Omnigrok<->Nanda bridge).
#       clamp     : every step, re-project ||W|| = rho*wc (hold the norm away from wc).
#                   Prediction: grok delayed/prevented (bidirectional control).
#
# Model: 2-layer MLP on (a+b) mod p (identical to run_grokfss.py), batched over seeds; manual AdamW.
# Rich data (criteria iv,v): full scalar trajectories incl. per-layer norms; embedding-matrix
# snapshots at ~8 checkpoints (Temp/, for deep Fourier/structural analysis); per-seed T_grok/T_mem;
# at-intervention norm before/after. Timestamped folder under D:\Colab-local; atomic writes; resume.
# Analyze with the accompanying logic (dose-response T_grok(rho); arm vs matched control).
# =====================================================================================
import argparse, json, math, os, time, hashlib, datetime, platform
import numpy as np
try:
    import torch
    import torch.nn.functional as F
except Exception:
    print("ERROR: PyTorch required."); raise

# critical norm by p (from the observational run; total ||W|| over [E,W1,W2]).
WC = {29: 42.1, 41: 47.6, 59: 54.6, 79: 61.2, 97: 67.1}

def default_grid():
    d, H = 128, 256
    p, alpha, lam, lr, seeds = 59, 0.40, 1.0, 1e-3, 16
    t_int, max_steps = 500, 20_000
    wc = WC[p]
    base = dict(optim="adamw", p=p, alpha=alpha, d_model=d, H=H, lam=lam, lr=lr,
                seeds=seeds, t_int=t_int, max_steps=max_steps, wc=wc)
    grid = []
    # control
    grid.append(dict(base, arm="none", rho=1.0, layer="all", f=1.0))
    # dose-response: rescale ALL to rho * wc
    for rho in [0.70, 0.85, 1.00, 1.15, 1.30]:
        grid.append(dict(base, arm="rescale", rho=rho, layer="all", f=1.0))
    # localization: scale one group by fixed f at t_int
    for layer in ["all", "embed", "hidden", "out"]:
        grid.append(dict(base, arm="layer", rho=1.0, layer=layer, f=0.80))
    # hold-high clamp (continuous): prevent the descent
    for rho in [0.90, 1.00, 1.15, 1.30]:
        grid.append(dict(base, arm="clamp", rho=rho, layer="all", f=1.0))
    return dedupe(grid)

def quick_grid():
    d, H = 32, 64
    base = dict(optim="adamw", p=17, alpha=0.5, d_model=d, H=H, lam=1.0, lr=2e-3,
                seeds=3, t_int=60, max_steps=1200, wc=18.0)
    return dedupe([dict(base, arm="none", rho=1.0, layer="all", f=1.0),
                   dict(base, arm="rescale", rho=0.85, layer="all", f=1.0),
                   dict(base, arm="layer", rho=1.0, layer="embed", f=0.8)])

def dedupe(grid):
    seen, uniq = set(), []
    for c in grid:
        cid = config_id(c)
        if cid not in seen: seen.add(cid); uniq.append(c)
    return uniq

def base_key(c):
    # seed/init/data depend ONLY on this (shared by control + all interventions)
    return f"{c['optim']}_p{c['p']}_a{c['alpha']:.2f}_d{c['d_model']}_H{c['H']}_wd{c['lam']:.0e}_lr{c['lr']:.0e}"

def config_id(c):
    arm = c["arm"]
    if arm == "none":    tag = "ctrl"
    elif arm == "rescale": tag = f"rescale_rho{c['rho']:.2f}"
    elif arm == "layer":   tag = f"layer_{c['layer']}_f{c['f']:.2f}"
    elif arm == "clamp":   tag = f"clamp_rho{c['rho']:.2f}"
    else: tag = arm
    return (base_key(c) + "__" + tag).replace("+", "")

def log_step_grid(max_steps, n_points=360):
    pts = np.unique(np.round(np.geomspace(1, max_steps, n_points)).astype(int))
    return np.unique(np.concatenate([[0], np.clip(pts, 0, max_steps)]))

def spec_entropy(M):
    sg = torch.linalg.svdvals(M); p = sg.pow(2); p = p / (p.sum(-1, keepdim=True) + 1e-30)
    return -(p * (p + 1e-30).log()).sum(-1)

def fourier_gini(E):
    S, p, d = E.shape
    F_ = torch.fft.rfft(E, dim=1); power = (F_.abs() ** 2).sum(-1)[:, 1:]
    power = power / (power.sum(-1, keepdim=True) + 1e-30)
    sp, _ = torch.sort(power, dim=-1); n = sp.shape[-1]
    idx = torch.arange(1, n + 1, device=E.device, dtype=E.dtype)
    return (2 * (idx * sp).sum(-1)) / (n * sp.sum(-1) + 1e-30) - (n + 1) / n

# -------------------------------------------------------------------------------------
def run_config(c, device, dtype=torch.float32, acc_mem=0.99, acc_grok=0.90):
    p, S, d, H = c["p"], c["seeds"], c["d_model"], c["H"]
    lam, lr = c["lam"], c["lr"]
    wc, rho, arm, layer, f, t_int = c["wc"], c["rho"], c["arm"], c["layer"], c["f"], c["t_int"]
    # CRUCIAL: seed from base_key (NOT config_id) -> matched counterfactual across arms
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
    E  = param(p, d,   scale=1.0 / math.sqrt(d))
    W1 = param(2 * d, H, scale=1.0 / math.sqrt(2 * d))
    b1 = param(H,      scale=1e-6)
    W2 = param(H, p,   scale=1.0 / math.sqrt(H))
    b2 = param(p,      scale=1e-6)
    params = [E, W1, b1, W2, b2]; pnames = ["E", "W1", "b1", "W2", "b2"]
    decay  = [True, True, False, True, False]
    wmats  = {"E": E, "W1": W1, "W2": W2}                 # weight matrices entering ||W||
    group  = {"all": ["E", "W1", "W2"], "embed": ["E"], "hidden": ["W1"], "out": ["W2"]}
    b1m, b2m, aeps = 0.9, 0.999, 1e-8
    ms = [torch.zeros_like(t) for t in params]
    vs = [torch.zeros_like(t) for t in params]

    def total_norm():
        return torch.sqrt(sum(wmats[k].pow(2).sum(dim=tuple(range(1, wmats[k].ndim))) for k in wmats))  # (S,)
    def group_norm(names):
        return torch.sqrt(sum(wmats[k].pow(2).sum(dim=tuple(range(1, wmats[k].ndim))) for k in names))

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

    def reset_moments():
        for i in range(len(params)): ms[i].zero_(); vs[i].zero_()

    def apply_rescale_all(target):                       # per-seed scale all wmats so ||W||=target
        scale = (target / (total_norm() + 1e-30))
        for k in wmats:
            sh = [S] + [1] * (wmats[k].ndim - 1)
            wmats[k].mul_(scale.view(*sh))
    def apply_layer_scale(names, factor):
        for k in names:
            wmats[k].mul_(factor)

    steps_to_log = set(int(s) for s in log_step_grid(c["max_steps"]))
    ckpt_steps = sorted(set(int(s) for s in np.unique(np.round(
                    np.geomspace(1, c["max_steps"], 8)).astype(int)) ) | {0, t_int})
    keys = ['train_loss','test_loss','train_acc','test_acc','weight_norm',
            'wn_E','wn_W1','wn_W2','spec_entropy_E','eff_rank_E','fourier_gini']
    logged, rec = [], {k: [] for k in keys}
    E_ckpts, E_ckpt_steps = [], []
    norm_at_int = dict(before=None, after=None)
    t0 = time.time()
    for step in range(c["max_steps"] + 1):
        # ---- intervention BEFORE the logging/step at t_int ----
        if step == t_int and arm in ("rescale", "layer"):
            with torch.no_grad():
                norm_at_int["before"] = total_norm().cpu().numpy()
                if arm == "rescale": apply_rescale_all(rho * wc)
                else:                apply_layer_scale(group[layer], f)
                norm_at_int["after"] = total_norm().cpu().numpy()
            # NOTE: do NOT reset AdamW moments (a reset makes vhat~0 -> exploding first step
            # that re-grows the norm and washes out the intervention). Keep optimizer state.
        if step in steps_to_log:
            with torch.no_grad():
                ce_tr, acc_tr = loss_acc(a_tr, b_tr, y_tr)
                ce_te, acc_te = loss_acc(a_te, b_te, y_te)
                wn = total_norm(); seE = spec_entropy(E); fg = fourier_gini(E)
                vals = [('train_loss',ce_tr),('test_loss',ce_te),('train_acc',acc_tr),
                        ('test_acc',acc_te),('weight_norm',wn),
                        ('wn_E',group_norm(["E"])),('wn_W1',group_norm(["W1"])),('wn_W2',group_norm(["W2"])),
                        ('spec_entropy_E',seE),('eff_rank_E',seE.exp()),('fourier_gini',fg)]
            logged.append(step)
            for k, v in vals: rec[k].append(v.cpu().numpy())
        if step in ckpt_steps:
            E_ckpts.append(E.detach().cpu().numpy().astype(np.float32)); E_ckpt_steps.append(step)
        if step == c["max_steps"]: break
        # ---- gradient step ----
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
            # ---- continuous clamp: hold ||W|| at rho*wc ----
            if arm == "clamp" and step + 1 >= t_int:
                apply_rescale_all(rho * wc)

    steps = np.asarray(logged, np.int64)
    out = {k: np.stack(v, 1).astype(np.float32) for k, v in rec.items()}
    tr_acc, te_acc = out['train_acc'], out['test_acc']
    Tmem = np.full(S, -1, np.int64); Tgrok = np.full(S, -1, np.int64)
    for s in range(S):
        im = np.where(tr_acc[s] >= acc_mem)[0]; ig = np.where(te_acc[s] >= acc_grok)[0]
        if len(im): Tmem[s] = steps[im[0]]
        if len(ig): Tgrok[s] = steps[ig[0]]
    delay = np.where((Tgrok > 0) & (Tmem >= 0), Tgrok - Tmem, -1).astype(np.int64)

    payload = dict(steps=steps, p=np.int64(p), alpha=np.float32(c["alpha"]),
                   d_model=np.int64(d), H=np.int64(H), lam=np.float32(lam), lr=np.float32(lr),
                   arm=str(arm), rho=np.float32(rho), layer=str(layer), f=np.float32(f),
                   t_int=np.int64(t_int), wc=np.float32(wc), max_steps=np.int64(c["max_steps"]),
                   n_train=np.int64(n_tr), n_test=np.int64(n_te),
                   T_mem_per_seed=Tmem, T_grok_per_seed=Tgrok, grok_delay_per_seed=delay,
                   norm_before_int=(norm_at_int["before"] if norm_at_int["before"] is not None
                                    else np.full(S, np.nan, np.float32)),
                   norm_after_int=(norm_at_int["after"] if norm_at_int["after"] is not None
                                   else np.full(S, np.nan, np.float32)), **out)
    # rich structural snapshots for deep analysis (kept in Temp/)
    snap = dict(E_ckpts=np.stack(E_ckpts, 0), E_ckpt_steps=np.asarray(E_ckpt_steps, np.int64),
                T_grok=Tgrok, T_mem=Tmem, arm=str(arm), rho=np.float32(rho),
                layer=str(layer), f=np.float32(f), wc=np.float32(wc),
                norm_at_grok=np.asarray([out['weight_norm'][s, np.argmin(np.abs(steps - Tgrok[s]))]
                                         if Tgrok[s] > 0 else np.nan for s in range(S)], np.float32))
    return payload, snap, time.time() - t0, (S, n_tr)

# atomic IO / checkpoint (same scheme as the other runners)
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
        root = os.path.join(args.out_root, f"paperA_normint_{ts}")
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
        row = dict(config_id=cid, arm=c["arm"], rho=c["rho"], layer=c["layer"], f=c["f"],
                   p=c["p"], alpha=c["alpha"], wc=c["wc"], t_int=c["t_int"], seeds=S,
                   grok_frac=float((tg > 0).mean()),
                   T_grok_median=(float(np.median(conv)) if len(conv) else None),
                   T_grok_mean=(float(conv.mean()) if len(conv) else None),
                   norm_at_grok_mean=float(np.nanmean(snap["norm_at_grok"]))
                                     if np.isfinite(snap["norm_at_grok"]).any() else None,
                   seconds=round(dt, 1))
        summary = [r for r in summary if r.get("config_id") != cid] + [row]
        atomic_write_json(spath, summary)
        ckpt[cid] = dict(done=True, seconds=round(dt, 1))
        atomic_write_json(os.path.join(root, "checkpoint.json"), ckpt)
        log_line(root, f"[{i+1}/{len(grid)}] {cid}  grok={int((tg>0).sum())}/{S}  "
                       f"Tgrok={row['T_grok_median']}  {dt:.1f}s")
    log_line(root, "ALL DONE")

if __name__ == "__main__":
    main()
