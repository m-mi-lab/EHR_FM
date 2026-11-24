"""Pretraining for every architecture: MLM for bidirectional models, next-token for causal.

Logs natively to MLflow (experiment ICLR-pretrain, one run per invocation):
params, per-epoch train/val loss, perplexity, LR, epoch time, best val, stop
reason, and history.json as an artifact. No log-tailing bridge.
"""
import argparse, json, math, os, time, torch, numpy as np, mlflow
from pathlib import Path
from torch.utils.data import DataLoader
from data import EICSequences, load_vocab, PREFIX_LEN, MASK_TOKEN
from model import build, ARCHS, CONTEXT, BATCH, ACCUM

# Mamba's selective-scan recurrence overflows in fp16 (it is trained in bf16
# upstream); every other model is fine in fp16. bf16 needs a GradScaler of
# enabled=False since it has fp32's exponent range.
AMP_DTYPE = {"mamba": torch.bfloat16, "mambas": torch.bfloat16, "mambal": torch.bfloat16,
             "bertl": torch.bfloat16, "bigbird": torch.bfloat16}   # the two deepest MLM models: fp16 overflowed (bertl) / collapsed (bigbird) at 3e-4
QUICK_VAL_WINDOWS = 1024   # fixed tuning-split subset scored every --val_every steps


def mask_tokens(code, mask, mask_id, p=0.15):
    """MLM corruption with a real, learnable [MASK] token. The demographic
    prefix (first PREFIX_LEN positions) is never masked and never predicted."""
    lab = code.clone()
    sel = (torch.rand_like(code, dtype=torch.float) < p) & (mask == 1)
    sel[:, :PREFIX_LEN] = False
    lab[~sel] = -100
    inp = code.clone(); inp[sel] = mask_id
    return inp, lab


def shift_tokens(code, mask):
    """Next-token targets, for causal models. Prefix positions are attended as
    context but carry no loss, as in ETHOS (y[:context_size] = -100)."""
    lab = code.clone()
    lab[:, :-1] = code[:, 1:]
    lab[:, -1] = -100
    lab[mask == 0] = -100
    lab[:, :PREFIX_LEN] = -100
    return code, lab


def _atomic_save(obj, path):
    """torch.save into <path>.tmp then rename: readers never see a half-written file, and a checkpoint that
    is still memory-mapped by this process keeps its old inode (truncating it in place raises SIGBUS)."""
    tmp = path.with_name(path.name + ".tmp"); torch.save(obj, tmp); os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bilstm", choices=sorted(ARCHS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_len", type=int, default=None, help="default: model.CONTEXT[arch]")
    ap.add_argument("--bs", type=int, default=None, help="micro-batch; default model.BATCH[arch]")
    ap.add_argument("--accum", type=int, default=None, help="grad-accum steps; default model.ACCUM[arch] (bs*accum*L = 32768)")
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2 (the roster uses 0.95)")
    ap.add_argument("--adam_eps", type=float, default=1e-8, help="AdamW eps (1e-6 = BERT setting)")
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=5)
    # Every model sees 32,768 tokens per optimizer step, so "step" is the same
    # token axis for all of them; epochs take 1-6 h, so MLflow gets in-epoch points.
    ap.add_argument("--log_every", type=int, default=200, help="optimizer steps between train_loss_step points")
    ap.add_argument("--val_every", type=int, default=1000, help="optimizer steps between val_loss_step points")
    ap.add_argument("--lr_patience", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", type=int, default=0,
                    help="continue from <out>/last.pt (model, optimizer, schedulers, scaler, RNG, epoch); "
                         "for hosts with a wall-clock limit (CHPC sessions)")
    a = ap.parse_args()
    if a.max_len is None: a.max_len = CONTEXT[a.model]
    if a.bs is None: a.bs = BATCH[a.model]
    if a.accum is None: a.accum = ACCUM[a.model]
    torch.manual_seed(a.seed); np.random.seed(a.seed)   # identical batch order for every model
    if a.out is None:
        a.out = f"{os.environ.get('ICLR_ROOT', Path(__file__).resolve().parents[2])}/data/models/{a.model}_pretrain"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    itos, stoi = load_vocab(); V = len(itos); MASK_ID = stoi[MASK_TOKEN]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # train: ETHOS-style random windows over the whole corpus; val: last window, fixed
    tr_ds = EICSequences("train", a.max_len, a.limit, windows="random", seed=a.seed)
    # persistent_workers MUST be False: workers are forked with a copy of the
    # dataset, so set_epoch() in the main process would never reach persistent
    # workers and every epoch would redraw epoch-0's windows. Re-forking 4
    # workers per epoch is free next to a 1-8 h epoch (the corpus is COW-shared).
    tr = DataLoader(tr_ds, batch_size=a.bs, shuffle=True, num_workers=int(os.environ.get("ICLR_WORKERS", 4)), drop_last=True,
                    persistent_workers=False)
    va = DataLoader(EICSequences("tuning", a.max_len, (a.limit // 8 if a.limit else None), windows="to_end"),
                    batch_size=a.bs, num_workers=min(2, int(os.environ.get("ICLR_WORKERS", 4))))
    qv = DataLoader(torch.utils.data.Subset(va.dataset, range(min(QUICK_VAL_WINDOWS, len(va.dataset)))),
                    batch_size=a.bs, num_workers=0)
    m = build(a.model, V, d=a.d, layers=a.layers, heads=a.heads, dropout=a.dropout).to(dev)
    causal = m.causal
    objective = ((lambda c, k, v: shift_tokens(c, k)) if causal
                 else (lambda c, k, v: mask_tokens(c, k, MASK_ID)))
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, a.beta2), eps=a.adam_eps)
    # Plateau LR, NOT cosine. With a 500-epoch cap a cosine schedule would still
    # be near peak LR when early stopping fires around epoch 50-100, so it would
    # never actually anneal. Halving on plateau anneals exactly when progress
    # stalls, which is what training to convergence requires.
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=a.lr_patience, min_lr=a.min_lr)
    warm = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min(1.0, (st + 1) / max(a.warmup, 1)))
    amp_dtype = AMP_DTYPE.get(a.model, torch.float16)
    if amp_dtype == torch.bfloat16 and dev == "cuda" and not torch.cuda.is_bf16_supported():
        a.amp = 0                                  # pre-Ampere GPU: mamba runs fp32 rather than fp16 (overflows)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(a.amp) and amp_dtype == torch.float16)
    lossf = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def evaluate(loader):
        """Mean loss over a loader. The MLM mask draw is re-seeded so the same
        positions are scored every time (a smooth curve, not mask noise); the
        RNG is restored afterwards so training draws are untouched."""
        m.eval(); vt = vn = 0
        with torch.no_grad(), torch.random.fork_rng(devices=[0] if dev == "cuda" else []):
            torch.manual_seed(a.seed)
            for b in loader:
                code, mask = b["code"].to(dev), b["mask"].to(dev)
                inp, lab = objective(code, mask, V)
                with torch.cuda.amp.autocast(enabled=bool(a.amp), dtype=amp_dtype):
                    vt += lossf(m.forward_mlm(inp, mask).float().reshape(-1, V), lab.reshape(-1)).item()
                vn += 1
        m.train()
        return vt / max(vn, 1)
    nparam = sum(q.numel() for q in m.parameters())

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
    mlflow.set_experiment("ICLR-pretrain")
    # GPU util / memory, CPU, RAM every 60 s (system/* metrics), sampled by mlflow's own thread
    mlflow.set_system_metrics_sampling_interval(int(os.environ.get("MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL", 60)))
    mlflow.start_run(run_name=a.model, log_system_metrics=True)
    mlflow.set_tags({"arch": a.model, "objective": "next-token" if causal else "MLM",
                     "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "?"), "stage": "pretrain"})
    mlflow.log_params({**vars(a), "params_M": round(nparam / 1e6, 1), "vocab": V,
                       "train_windows_per_epoch": len(tr.dataset), "train_tokens": tr_ds.c.n_tokens,
                       "val_n": len(va.dataset), "objective": "next-token" if causal else "MLM",
                       "window_sampling": "random(ETHOS)", "prefix_len": PREFIX_LEN})
    print(f"arch={a.model} causal={causal} objective={'next-token' if causal else 'MLM'} "
          f"mlflow_run={mlflow.active_run().info.run_id[:8]}", flush=True)
    print(f"vocab={V} train_windows={len(tr.dataset):,} train_tokens={tr_ds.c.n_tokens:,} val={len(va.dataset):,} device={dev} "
          f"params={nparam/1e6:.1f}M d={a.d} L={a.layers} H={a.heads} ctx={a.max_len} bs={a.bs}x{a.accum} "
          f"amp={bool(a.amp)}:{str(amp_dtype).split('.')[-1]} dropout={a.dropout} wd={a.wd}", flush=True)

    best, bad, hist, gstep, stop = 1e9, 0, [], 0, "epoch cap reached"
    start_ep = 0
    if a.resume and ((out / "last.pt").exists() or ((out / "best.pt").exists() and (out / "history.json").exists())):
        ck = None
        if (out / "last.pt").exists():
            try:   # mmap: no private copy of the 8.6 GB gpt2l checkpoint during resume
                ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False, mmap=True)
            except Exception as e:   # truncated by a crash mid-write: continue from best.pt + history.json instead of dying
                print(f"last.pt unreadable ({type(e).__name__}: {e}); falling back to best.pt + history.json", flush=True)
        else: print("no last.pt; resuming from best.pt + history.json (rewound run)", flush=True)
        if ck is not None:
            m.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
            warm.load_state_dict(ck["warm"]); scaler.load_state_dict(ck["scaler"])
            for st in opt.state.values():   # detach anything the optimizer kept from the mapping (e.g. `step`)
                for k, v in list(st.items()):
                    if torch.is_tensor(v) and v.device.type == "cpu": st[k] = v.clone()
            best, bad, hist, gstep, start_ep = ck["best"], ck["bad"], list(ck["hist"]), ck["gstep"], ck["epoch"] + 1
            torch.set_rng_state(ck["rng_cpu"].clone()); np.random.set_state(ck["rng_np"])
            if dev == "cuda": torch.cuda.set_rng_state(ck["rng_cuda"].clone())
            mlflow.log_params({"resumed_from_epoch": ck["epoch"]})
            print(f"RESUMED from epoch {ck['epoch']} (best {best:.4f}, gstep {gstep}, lr {opt.param_groups[0]['lr']:.2e})", flush=True)
            del ck   # drop the memory-mapped checkpoint before last.pt is rewritten at the end of the epoch
        elif (out / "best.pt").exists() and (out / "history.json").exists():
            bk = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
            m.load_state_dict(bk["model"], strict=True); del bk
            hist = json.loads((out / "history.json").read_text())
            last = hist[-1]; best = min(h["val_loss"] for h in hist); start_ep = last["epoch"] + 1
            gstep = start_ep * (len(tr) // a.accum)          # optimizer steps so far (same step axis as before)
            for g in opt.param_groups: g["lr"] = last["lr"]  # keep the schedule's current LR; plateau counters restart
            mlflow.log_params({"resumed_from_epoch": last["epoch"], "resumed_from_best_pt": 1})
            print(f"RESUMED from best.pt at epoch {last['epoch']} (best {best:.4f}, lr {last['lr']:.2e}); "
                  f"optimizer moments and plateau counters restarted, weights are the best epoch's", flush=True)
        else:
            raise RuntimeError("last.pt is unreadable and no best.pt/history.json to fall back on")
    try:
        for ep in range(start_ep, a.epochs):
            m.train(); t0 = time.time(); tot = n = 0
            tr_ds.set_epoch(ep)                      # new window draws, identical across archs
            opt.zero_grad(set_to_none=True); micro = 0
            win = 0.0; win_n = 0; gn_win = 0.0; gn_n = 0; skip_win = 0; tok_win = 0; t_win = time.time()
            for b in tr:
                code, mask = b["code"].to(dev), b["mask"].to(dev)
                inp, lab = objective(code, mask, V)
                with torch.cuda.amp.autocast(enabled=bool(a.amp), dtype=amp_dtype):
                    logits = m.forward_mlm(inp, mask)
                    loss = lossf(logits.float().reshape(-1, V), lab.reshape(-1))
                if not torch.isfinite(loss):
                    # abort loudly: a NaN run otherwise trains on garbage for hours
                    raise RuntimeError(f"non-finite loss at epoch {ep} step {n} (arch={a.model}, amp={amp_dtype})")
                scaler.scale(loss / a.accum).backward()
                tot += loss.item(); n += 1; micro += 1; win += loss.item(); win_n += 1; tok_win += int(mask.sum())
                if micro % a.accum == 0:              # one optimizer step per 32,768 tokens
                    scaler.unscale_(opt)
                    gn = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0))
                    if math.isfinite(gn): gn_win += gn; gn_n += 1
                    else: skip_win += 1                 # fp16 overflow: GradScaler skips this step and halves the scale
                    scaler.step(opt); scaler.update()
                    opt.zero_grad(set_to_none=True)
                    if gstep < a.warmup: warm.step()   # plateau owns the LR after warmup
                    gstep += 1
                    if gstep % a.log_every == 0:      # in-epoch train curve
                        now = time.time()
                        # divide by what the window actually held: the first window of an
                        # epoch is short when the epoch boundary is not a multiple of log_every
                        live = {"train_loss_step": win / max(win_n, 1),
                                "train_ppl_step": float(np.exp(win / max(win_n, 1))),
                                "lr_step": opt.param_groups[0]["lr"],
                                "grad_norm": gn_win / max(gn_n, 1),         # pre-clip total norm, mean over finite steps
                                "amp_skipped_steps": skip_win,              # overflowed (skipped) optimizer steps in window
                                "tokens_per_sec": tok_win / max(now - t_win, 1e-6),   # real (non-pad) tokens
                                "epoch_progress": ep + micro / len(tr)}
                        if scaler.is_enabled(): live["amp_scale"] = scaler.get_scale()
                        mlflow.log_metrics(live, step=gstep)
                        win = 0.0; win_n = 0; gn_win = 0.0; gn_n = 0; skip_win = 0; tok_win = 0; t_win = now
                    if gstep % a.val_every == 0:      # in-epoch val curve (fixed 1,024-window subset)
                        qvl = evaluate(qv)
                        mlflow.log_metrics({"val_loss_step": qvl, "val_ppl_step": float(np.exp(qvl))}, step=gstep)
            trl, val = tot / max(n, 1), evaluate(va)   # full tuning split decides LR / stopping
            sched.step(val)
            cur_lr = opt.param_groups[0]["lr"]; secs = time.time() - t0
            hist.append({"epoch": ep, "train_loss": trl, "val_loss": val,
                         "val_ppl": float(np.exp(val)), "lr": cur_lr, "secs": secs})
            mlflow.log_metrics({"train_loss": trl, "train_ppl": float(np.exp(trl)), "val_loss": val,
                                "val_ppl": float(np.exp(val)), "lr": cur_lr, "epoch_secs": secs,
                                "epoch": ep}, step=gstep)   # same step axis as *_step
            print(f"epoch={ep} train_loss={trl:.4f} val_loss={val:.4f} val_ppl={np.exp(val):.2f} "
                  f"lr={cur_lr:.2e} ({secs:.0f}s)", flush=True)
            (out / "history.json").write_text(json.dumps(hist, indent=2))
            mlflow.log_artifact(str(out / "history.json"))    # visible while the run is still going
            if val < best - 1e-4:
                best, bad = val, 0
                _atomic_save({"model": m.state_dict(), "vocab": V, "d": a.d, "layers": a.layers,
                              "heads": a.heads, "arch": a.model}, out / "best.pt")
                mlflow.log_metric("best_val_loss", best, step=gstep)
                print(f"  saved best val_loss={best:.4f}", flush=True)
            else:
                bad += 1
                if bad >= a.patience:
                    stop = f"no improvement for {a.patience} epochs (best {best:.4f} at epoch {ep - bad})"
                    break
            if opt.param_groups[0]["lr"] <= a.min_lr * 1.01 and bad >= a.lr_patience:
                stop = f"LR floor {a.min_lr:.1e} reached with no recent improvement (best {best:.4f})"
                break
            # full training state, so a wall-clock-limited host can continue with --resume 1
            _atomic_save({"model": m.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                          "warm": warm.state_dict(), "scaler": scaler.state_dict(), "best": best, "bad": bad,
                          "hist": hist, "gstep": gstep, "epoch": ep, "rng_cpu": torch.get_rng_state(),
                          "rng_np": np.random.get_state(),
                          "rng_cuda": torch.cuda.get_rng_state() if dev == "cuda" else None}, out / "last.pt")
    except BaseException as e:                       # includes KeyboardInterrupt
        mlflow.set_tag("stop_reason", f"exception: {type(e).__name__}")
        mlflow.end_run(status="FAILED"); raise
    print(f"CONVERGED: {stop}", flush=True)
    mlflow.set_tag("stop_reason", stop)
    mlflow.log_metrics({"final_best_val_loss": best, "final_best_val_ppl": float(np.exp(best)),
                        "epochs_run": len(hist)})
    mlflow.log_artifact(str(out / "history.json"))
    mlflow.end_run(status="FINISHED")
    print("PRETRAIN DONE best_val_loss=%.4f" % best, flush=True)


if __name__ == "__main__": main()
