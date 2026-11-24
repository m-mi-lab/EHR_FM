"""Per-task finetuning from a pretrained checkpoint (any architecture).

Logs natively to MLflow (experiment ICLR-finetune, run name <arch>/<task>).
"""
import argparse, json, os, time, numpy as np, torch, mlflow
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from data import EICTask, load_vocab
from model import build, CONTEXT
from pretrain import AMP_DTYPE


def evaluate(m, dl, dev):
    m.eval(); ys, ps = [], []
    with torch.no_grad():
        for b in dl:
            p = torch.sigmoid(m.forward_clf(b["code"].to(dev), b["mask"].to(dev)))
            ys.append(b["label"].numpy()); ps.append(p.cpu().numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return (float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
            float(average_precision_score(y, p)), y, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", default="bilstm")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_len", type=int, default=None, help="default: model.CONTEXT[arch]")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--accum", type=int, default=1,
                    help="gradient-accumulation steps; effective batch = bs*accum (the baselines use 64x1)")
    # 5e-5, not 5e-4: standard for finetuning a ~90M transformer. 5e-4 was
    # tolerable at d=256 but is 10x the norm at base geometry and risks
    # destroying the pretrained representation in the first epoch.
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    if a.max_len is None: a.max_len = CONTEXT[a.model]
    root = Path(os.environ.get("ICLR_ROOT", Path(__file__).resolve().parents[2])) / "data" / "models"
    if a.ckpt is None: a.ckpt = str(root / f"{a.model}_pretrain" / "best.pt")
    if a.out is None: a.out = str(root / f"{a.model}_ft")
    out = Path(a.out) / a.task; out.mkdir(parents=True, exist_ok=True)
    itos, _ = load_vocab(); V = len(itos)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = DataLoader(EICTask(a.task, "train", a.max_len, a.limit), batch_size=a.bs, shuffle=True, num_workers=2, drop_last=True)
    va = DataLoader(EICTask(a.task, "tuning", a.max_len, (a.limit // 8 if a.limit else None)), batch_size=a.bs, num_workers=0)
    te = DataLoader(EICTask(a.task, "held_out", a.max_len, (a.limit // 8 if a.limit else None)), batch_size=a.bs, num_workers=0)
    ck = torch.load(a.ckpt, map_location="cpu")
    m = build(ck.get("arch", a.model), V, d=ck["d"], layers=ck.get("layers", 4), heads=ck.get("heads", 8)).to(dev)
    # strict=True: a silent partial load (strict=False) would finetune a
    # randomly initialised backbone while reporting success
    m.load_state_dict(ck["model"], strict=True)
    pos = float(np.mean(tr.dataset.y)); w = torch.tensor((1 - pos) / max(pos, 1e-6), device=dev)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=w)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
    amp_dtype = AMP_DTYPE.get(a.model, torch.float16)   # same dtype as the pretrain: BigBird collapses in fp16
    scaler = torch.cuda.amp.GradScaler(enabled=bool(a.amp) and amp_dtype == torch.float16)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
    mlflow.set_experiment("ICLR-finetune")
    mlflow.start_run(run_name=f"{a.model}/{a.task}")
    mlflow.set_tags({"arch": ck.get("arch", a.model), "task": a.task, "stage": "finetune",
                     "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "?")})
    mlflow.log_params({**vars(a), "train_n": len(tr.dataset), "val_n": len(va.dataset),
                       "test_n": len(te.dataset), "prevalence": round(pos, 4),
                       "pretrain_d": ck["d"], "pretrain_layers": ck.get("layers"), "pretrain_heads": ck.get("heads")})
    print(f"task={a.task} arch={ck.get('arch', a.model)} train={len(tr.dataset):,} val={len(va.dataset):,} "
          f"test={len(te.dataset):,} prev={pos:.4f} mlflow_run={mlflow.active_run().info.run_id[:8]}", flush=True)

    best, bad, hist = -1, 0, []
    try:
        for ep in range(a.epochs):
            m.train(); t0 = time.time(); opt.zero_grad(set_to_none=True)
            for i, b in enumerate(tr, 1):
                with torch.cuda.amp.autocast(enabled=bool(a.amp), dtype=amp_dtype):
                    logit = m.forward_clf(b["code"].to(dev), b["mask"].to(dev))
                loss = lossf(logit.float(), b["label"].to(dev)) / a.accum
                scaler.scale(loss).backward()
                if i % a.accum == 0:  # one optimizer step per bs*accum sequences
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            auc, ap_, _, _ = evaluate(m, va, dev); secs = time.time() - t0
            hist.append({"epoch": ep, "val_auroc": auc, "val_auprc": ap_, "secs": secs})
            mlflow.log_metrics({"val_auroc": auc, "val_auprc": ap_, "epoch_secs": secs}, step=ep)
            print(f"epoch={ep} val_auroc={auc:.4f} val_auprc={ap_:.4f} ({secs:.0f}s)", flush=True)
            if auc > best + 1e-4:
                best, bad = auc, 0
                torch.save({"model": m.state_dict(), "vocab": V, "d": ck["d"], "layers": ck.get("layers", 4),
                            "heads": ck.get("heads", 8), "arch": ck.get("arch", a.model)}, out / "best.pt")
                mlflow.log_metric("best_val_auroc", best, step=ep)
            else:
                bad += 1
                if bad >= a.patience: print("early stop", flush=True); break
        ck2 = torch.load(out / "best.pt", map_location="cpu"); m.load_state_dict(ck2["model"], strict=True)
        auc, ap_, y, p = evaluate(m, te, dev)
    except BaseException as e:
        mlflow.set_tag("stop_reason", f"exception: {type(e).__name__}")
        mlflow.end_run(status="FAILED"); raise
    res = {"task": a.task, "model": ck.get("arch", a.model), "test_auroc": auc, "test_auprc": ap_,
           "n_test": int(len(y)), "prevalence": float(y.mean()), "val_best_auroc": best, "history": hist}
    (out / "result.json").write_text(json.dumps(res, indent=2))
    np.savez_compressed(out / "test_preds.npz", y=y.astype(np.int8), p=p.astype(np.float32))
    mlflow.log_metrics({"test_auroc": auc, "test_auprc": ap_, "val_best_auroc": best, "epochs_run": len(hist)})
    mlflow.log_artifact(str(out / "result.json"))
    mlflow.end_run(status="FINISHED")
    print(f"RESULT {a.task}: test_auroc={auc:.4f} test_auprc={ap_:.4f} n={len(y):,}", flush=True)


if __name__ == "__main__": main()
