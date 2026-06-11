#!/usr/bin/env python3
# =====================================================================================
# run_tfnormint.py  —  DIFFERENTIAL causal test of the critical-norm hypothesis on a Transformer.
#
# Motivation (from the tfgrok analysis): on a LayerNorm Transformer the TOTAL weight norm is a
# meaningless mixture; the functionally relevant norm is the UNEMBEDDING ||U|| (logits = LN(x)U, and
# LN(x) has unit RMS, so ||U|| is literally the logit inverse-temperature). The unembedding norm at
# grokking is sharply concentrated (CV 0.03-0.05 across alpha, 0.04 across lr) and scales as p^0.77 ---
# it replicates the MLP critical-norm on the correct variable. Upstream groups (embed/attn/mlp) sit
# before LayerNorms and are functionally washed out, so their norms drift.
#
# DIFFERENTIAL design (built-in negative control, best practice):
#   * Clamp ||U|| (functional)  -> PREDICT gating: a monotone dose-response (below wc => faster,
#     above => delayed/prevented), mirroring the MLP total-norm clamp.
#   * Clamp ||mlp|| or ||embed|| (LN-washed) at the same relative level -> PREDICT a much WEAKER effect.
#   The contrast rules out the confound "any norm perturbation changes grokking" and tests the
#   LayerNorm/functional-norm mechanism directly.
#
# Matched counterfactual: the seed depends only on the base config (NOT the arm), so control and every
# clamp share identical init, data, and pre-clamp trajectory. Continuous clamp, NO optimizer-moment reset
# (a reset spikes the first post-clamp step and washes out the intervention -- learned on the MLP runner).
# Model is identical to run_tfgrok.py. Rich data + Temp/ snapshots; timestamped under D:\Colab-local; resume.
# =====================================================================================
import argparse, json, math, os, time, hashlib, datetime, platform
import numpy as np
try:
    import torch
    import torch.nn.functional as F
except Exception:
    print("ERROR: PyTorch required."); raise

# per-group critical norms at the reference config p=71, alpha=0.40, lam=1 (measured in tfgrok run).
WC_GROUP = {"unembed": 12.1, "mlp": 11.2, "embed": 3.4, "attn": 7.2}

def default_grid():
    d, nh, H = 128, 4, 512
    base = dict(optim="adamw", p=71, alpha=0.40, d_model=d, n_heads=nh, n_layers=1, H=H,
                lam=1.0, lr=1e-3, seeds=16, t_int=600, max_steps=25_000, wc=dict(WC_GROUP))
    grid = [dict(base, arm="none", group="none", rho=1.0)]
    for rho in [0.85, 1.00, 1.15, 1.30]:                  # functional norm: dose-response (predict gating)
        grid.append(dict(base, arm="clamp", group="unembed", rho=rho))
    for rho in [0.85, 1.30]:                              # LN-washed control: predict weak effect
        grid.append(dict(base, arm="clamp", group="mlp", rho=rho))
    grid.append(dict(base, arm="clamp", group="embed", rho=1.30))   # second negative control
    return dedupe(grid)

def quick_grid():
    base = dict(optim="adamw", p=17, alpha=0.5, d_model=32, n_heads=2, n_layers=1, H=64,
                lam=1.0, lr=2e-3, seeds=3, t_int=60, max_steps=1500,
                wc=dict(unembed=6.0, mlp=5.0, embed=2.0, attn=3.0))
    return dedupe([dict(base, arm="none", group="none", rho=1.0),
                   dict(base, arm="clamp", group="unembed", rho=0.85),
                   dict(base, arm="clamp", group="mlp", rho=1.30)])

def dedupe(grid):
    seen, uniq = set(), []
    for c in grid:
        k = config_id(c)
        if k not in seen: seen.add(k); uniq.append(c)
    return uniq

def base_key(c):                                          # seed/init/data depend ONLY on this
    return (f"{c['optim']}_p{c['p']}_a{c['alpha']:.2f}_d{c['d_model']}_h{c['n_heads']}"
            f"_H{c['H']}_wd{c['lam']:.0e}_lr{c['lr']:.0e}")

def config_id(c):
    tag = "ctrl" if c["arm"] == "none" else f"clamp_{c['group']}_rho{c['rho']:.2f}"
    return base_key(c) + "__" + tag

def log_step_grid(max_steps, n_points=360):
    pts = np.unique(np.round(np.geomspace(1, max_steps, n_points)).astype(int))
    return np.unique(np.concatenate([[0], np.clip(pts, 0, max_steps)]))

def spec_entropy(M):
    sg = torch.linalg.svdvals(M); pw = sg.pow(2); pw = pw / (pw.sum(-1, keepdim=True) + 1e-30)
    return -(pw * (pw + 1e-30).log()).sum(-1)
def fourier_gini(E):
    S, p, d = E.shape
    Fc = torch.fft.rfft(E, dim=1); power = (Fc.abs()**2).sum(-1)[:, 1:]
    power = power / (power.sum(-1, keepdim=True) + 1e-30)
    sp, _ = torch.sort(power, dim=-1); n = sp.shape[-1]
    idx = torch.arange(1, n+1, device=E.device, dtype=E.dtype)
    return (2*(idx*sp).sum(-1)) / (n*sp.sum(-1) + 1e-30) - (n+1)/n

def run_config(c, device, dtype=torch.float32, acc_mem=0.99, acc_grok=0.90):
    p, S = c["p"], c["seeds"]; d, nh, H = c["d_model"], c["n_heads"], c["H"]; dh = d // nh
    V = p + 1; lam, lr = c["lam"], c["lr"]
    arm, grp, rho, t_int, wc = c["arm"], c["group"], c["rho"], c["t_int"], c["wc"]
    base_seed = int(hashlib.sha1(base_key(c).encode()).hexdigest(), 16) % (2**31)   # matched counterfactual
    g = torch.Generator(device=device).manual_seed(base_seed)

    a_all = torch.arange(p, device=device).repeat_interleave(p)
    b_all = torch.arange(p, device=device).repeat(p)
    y_all = (a_all + b_all) % p
    eq = torch.full_like(a_all, p); toks_all = torch.stack([a_all, b_all, eq], 1)
    M = p*p; n_tr = max(2, int(round(c["alpha"]*M))); n_te = M - n_tr
    if n_te < 1: n_tr = M-1; n_te = 1
    perm = torch.stack([torch.randperm(M, generator=g, device=device) for _ in range(S)])
    tok_tr = toks_all[perm[:, :n_tr]]; y_tr = y_all[perm[:, :n_tr]]
    tok_te = toks_all[perm[:, n_tr:]]; y_te = y_all[perm[:, n_tr:]]

    def param(*shape, scale):
        return (scale*torch.randn(S, *shape, generator=g, device=device, dtype=dtype)).requires_grad_(True)
    E   = param(V, d, scale=1/math.sqrt(d)); pos = param(3, d, scale=0.1/math.sqrt(d))
    Wq  = param(d, d, scale=1/math.sqrt(d)); Wk = param(d, d, scale=1/math.sqrt(d))
    Wv  = param(d, d, scale=1/math.sqrt(d)); Wo = param(d, d, scale=1/math.sqrt(d))
    Win = param(d, H, scale=1/math.sqrt(d)); Wout = param(H, d, scale=1/math.sqrt(H))
    U   = param(d, V, scale=1/math.sqrt(d))
    params = [E, pos, Wq, Wk, Wv, Wo, Win, Wout, U]
    pnames = ["E","pos","Wq","Wk","Wv","Wo","Win","Wout","U"]
    norm_set = list(pnames)
    b1m, b2m, aeps = 0.9, 0.999, 1e-8
    ms = [torch.zeros_like(t) for t in params]; vs = [torch.zeros_like(t) for t in params]
    pdict = {nm: t for nm, t in zip(pnames, params)}
    groups = {"embed":["E","pos"], "attn":["Wq","Wk","Wv","Wo"], "mlp":["Win","Wout"], "unembed":["U"]}

    def ln(x):
        return (x - x.mean(-1, keepdim=True)) / (x.var(-1, unbiased=False, keepdim=True) + 1e-5).sqrt()
    def logits_for(tok):
        B = tok.shape[1]
        idx = tok.reshape(S, B*3, 1).expand(S, B*3, d)
        emb = torch.gather(E, 1, idx).reshape(S, B, 3, d)
        x = emb + pos.unsqueeze(1); xln = ln(x)
        qd = torch.einsum('sbd,sde->sbe', xln[:, :, -1, :], Wq).reshape(S, B, nh, dh)
        kd = torch.einsum('sbtd,sde->sbte', xln, Wk).reshape(S, B, 3, nh, dh)
        vd = torch.einsum('sbtd,sde->sbte', xln, Wv).reshape(S, B, 3, nh, dh)
        scores = torch.einsum('sbhe,sbthe->sbht', qd, kd) / math.sqrt(dh)
        attn = torch.softmax(scores, -1)
        z = torch.einsum('sbht,sbthe->sbhe', attn, vd).reshape(S, B, d)
        x_last = x[:, :, -1, :] + torch.einsum('sbd,sde->sbe', z, Wo)
        h = F.gelu(torch.einsum('sbd,sdm->sbm', ln(x_last), Win))
        x_last = x_last + torch.einsum('sbm,smd->sbd', h, Wout)
        return torch.einsum('sbd,sdv->sbv', ln(x_last), U)
    def loss_acc(tok, y):
        lg = logits_for(tok); logp = F.log_softmax(lg, -1)
        ce = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1).mean(1)
        acc = (lg.argmax(-1) == y).float().mean(1)
        return ce, acc
    def group_norm(names):
        return torch.sqrt(sum(pdict[k].pow(2).sum(dim=tuple(range(1, pdict[k].ndim))) for k in names))
    def weight_norm(): return group_norm(norm_set)
    def clamp_group(name, target):                         # project only `name` group to ||.||=target
        cur = group_norm(groups[name]); scale = target / (cur + 1e-30)
        for k in groups[name]:
            sh = [S] + [1]*(pdict[k].ndim - 1); pdict[k].mul_(scale.view(*sh))

    steps_to_log = set(int(s) for s in log_step_grid(c["max_steps"]))
    ckpt_steps = sorted(set(int(s) for s in np.unique(np.round(np.geomspace(1, c["max_steps"], 8)).astype(int))) | {0, t_int})
    keys = ['train_loss','test_loss','train_acc','test_acc','weight_norm',
            'wn_embed','wn_attn','wn_mlp','wn_unembed','fourier_gini','eff_rank_E']
    logged, rec = [], {k: [] for k in keys}
    E_ckpts, U_ckpts, ck_steps = [], [], []
    norm_at_int = dict(before=np.full(S, np.nan, np.float32), after=np.full(S, np.nan, np.float32))
    t0 = time.time()
    for step in range(c["max_steps"] + 1):
        if step in steps_to_log:
            with torch.no_grad():
                ce_tr, acc_tr = loss_acc(tok_tr, y_tr); ce_te, acc_te = loss_acc(tok_te, y_te)
                seE = spec_entropy(E[:, :p, :]); fg = fourier_gini(E[:, :p, :])
                vals = [('train_loss',ce_tr),('test_loss',ce_te),('train_acc',acc_tr),('test_acc',acc_te),
                        ('weight_norm',weight_norm()),('wn_embed',group_norm(groups["embed"])),
                        ('wn_attn',group_norm(groups["attn"])),('wn_mlp',group_norm(groups["mlp"])),
                        ('wn_unembed',group_norm(groups["unembed"])),('fourier_gini',fg),('eff_rank_E',seE.exp())]
            logged.append(step)
            for k, v in vals: rec[k].append(v.cpu().numpy())
        if step in ckpt_steps:
            E_ckpts.append(E.detach().cpu().numpy().astype(np.float32))
            U_ckpts.append(U.detach().cpu().numpy().astype(np.float32)); ck_steps.append(step)
        if step == c["max_steps"]: break
        ce_tr, _ = loss_acc(tok_tr, y_tr); loss = ce_tr.sum()
        for t in params: t.grad = None
        loss.backward()
        with torch.no_grad():
            t_ = step + 1
            for i, t in enumerate(params):
                t.mul_(1 - lr*lam)
                ms[i].mul_(b1m).add_(t.grad, alpha=1-b1m)
                vs[i].mul_(b2m).addcmul_(t.grad, t.grad, value=1-b2m)
                mhat = ms[i]/(1-b1m**t_); vhat = vs[i]/(1-b2m**t_)
                t.addcdiv_(mhat, vhat.sqrt().add_(aeps), value=-lr)
            if arm == "clamp" and step + 1 >= t_int:                 # continuous clamp, no moment reset
                if step + 1 == t_int: norm_at_int["before"] = group_norm(groups[grp]).cpu().numpy()
                clamp_group(grp, rho * wc[grp])
                if step + 1 == t_int: norm_at_int["after"] = group_norm(groups[grp]).cpu().numpy()

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
                   d_model=np.int64(d), n_heads=np.int64(nh), H=np.int64(H),
                   lam=np.float32(lam), lr=np.float32(lr), arm=str(arm), group=str(grp),
                   rho=np.float32(rho), t_int=np.int64(t_int), wc_group=np.float32(wc.get(grp, np.nan)),
                   max_steps=np.int64(c["max_steps"]), n_train=np.int64(n_tr), n_test=np.int64(n_te),
                   T_mem_per_seed=Tmem, T_grok_per_seed=Tgrok, grok_delay_per_seed=delay,
                   norm_before_int=norm_at_int["before"], norm_after_int=norm_at_int["after"], **out)
    snap = dict(E_ckpts=np.stack(E_ckpts, 0), U_ckpts=np.stack(U_ckpts, 0),
                ckpt_steps=np.asarray(ck_steps, np.int64), T_grok=Tgrok, T_mem=Tmem,
                arm=str(arm), group=str(grp), rho=np.float32(rho))
    return payload, snap, time.time() - t0, (S, n_tr)

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
    root = args.resume or os.path.join(args.out_root,
            f"paperA_tfnormint_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(os.path.join(root, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(root, "Temp"), exist_ok=True)
    print(f"configs: {len(grid)} | device: {device}")
    if args.dry_run:
        for c in grid: print("  ", config_id(c), "| seed_key:", base_key(c))
        return
    atomic_write_json(os.path.join(root, "grid_spec.json"),
                      [dict(config_id=config_id(c), **{k: v for k, v in c.items() if k != "wc"}) for c in grid])
    ckpt = load_checkpoint(root)
    log_line(root, f"start: {len(grid)} configs, device={device}, host={platform.node()}, "
                   f"resume={'yes' if args.resume else 'no'}, done={sum(1 for v in ckpt.values() if v.get('done'))}")
    spath = os.path.join(root, "results_summary.json")
    summary = json.load(open(spath)) if os.path.exists(spath) else []
    for i, c in enumerate(grid):
        cid = config_id(c)
        if ckpt.get(cid, {}).get("done"): continue
        payload, snap, dt, (S, n_tr) = run_config(c, device)
        atomic_savez(os.path.join(root, "metrics", cid + ".npz"), **payload)
        atomic_savez(os.path.join(root, "Temp", cid + "_struct.npz"), **snap)
        tg = payload["T_grok_per_seed"]; conv = tg[tg > 0]
        row = dict(config_id=cid, arm=c["arm"], group=c["group"], rho=c["rho"], p=c["p"],
                   alpha=c["alpha"], seeds=S, grok_frac=float((tg > 0).mean()),
                   T_grok_median=(float(np.median(conv)) if len(conv) else None),
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
