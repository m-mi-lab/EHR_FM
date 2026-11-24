"""Figures. Model-agnostic: discovers whatever models have results.

Layout it expects:
  data/models/<model>_pretrain/history.json
  data/models/<model>_ft/<task>/result.json
"""
import json, glob, os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(os.environ.get("ICLR_ROOT", Path(__file__).resolve().parents[2])) / "data"
RES, MOD = ROOT/"results", ROOT/"models"
FIG = RES/"figures"; FIG.mkdir(parents=True, exist_ok=True)

# fixed hue per model, assigned in order and never cycled
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"]
INK, INK2, RULE, GRID = "#0b0b0b", "#52514e", "#b8b8b4", "#e6e6e3"

def _save(fig, stem, family_prefix=None):
    """Write png+pdf, and delete superseded versions of the same figure.

    Aggregate figures are named for the models they contain, so adding a model
    changes the filename. Without this, pretrain_bilstm, pretrain_bert_bilstm
    and pretrain_bert_bilstm_gpt2 all survive and it is not obvious which is
    current."""
    if family_prefix:
        for old in FIG.glob(f"{family_prefix}*"):
            if old.stem != stem:
                old.unlink()
    for ext in ("png", "pdf"):
        fig.savefig(FIG/f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _suffix(models):
    """Aggregate figures are named for the models they actually contain, so a
    file called pretrain_bilstm never silently means 'all models'."""
    ms = sorted(models)
    return "_".join(ms) if 0 < len(ms) <= 3 else ("all" if ms else "none")


def _style(ax):
    ax.set_axisbelow(True); ax.grid(False)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    for sp in ("left","bottom"): ax.spines[sp].set_color(RULE); ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9, length=3, width=0.8, direction="out")

def color_of(model, all_models):
    return PALETTE[all_models.index(model) % len(PALETTE)]

# ---------- finetuned tasks, grouped by model ----------
def fig_tasks():
    res = {}
    for f in sorted(glob.glob(str(MOD/"*_ft/*/result.json"))):
        r = json.loads(Path(f).read_text())
        m = r.get("model") or Path(f).parents[1].name.replace("_ft","")
        res.setdefault(m, {})[r["task"]] = r
    if not res: print("no finetune results"); return
    ms = sorted(res); tasks = sorted({t for v in res.values() for t in v})
    fig, ax = plt.subplots(figsize=(1.9*len(tasks)*max(len(ms),1)+2, 3.6))
    w = 0.8/len(ms)
    for i, m in enumerate(ms):
        xs = np.arange(len(tasks)) + i*w - 0.4 + w/2
        ys = [res[m].get(t, {}).get("test_auroc", np.nan) for t in tasks]
        ax.bar(xs, ys, width=w*0.9, color=color_of(m, ms), label=m)
        for x, y in zip(xs, ys):
            if not np.isnan(y): ax.text(x, y+0.01, f"{y:.3f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(tasks))); ax.set_xticklabels(tasks, fontsize=9)
    ax.axhline(0.5, color=RULE, lw=0.8); ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("held-out AUROC", fontsize=10, color=INK2)
    if len(ms) > 1: ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    ax.set_title("Finetuned tasks", color=INK, fontsize=11, fontweight="bold", loc="left"); _style(ax)
    fig.tight_layout()
    sfx = _suffix(ms)
    _save(fig, f"tasks_{sfx}", "tasks_"); print(f"wrote tasks_{sfx}")

# ---------- pretraining curves, all models ----------
def fig_pretrain():
    hs = sorted(glob.glob(str(MOD/"*_pretrain/history.json")))
    if not hs: print("no pretrain history"); return
    ms = [Path(h).parent.name.replace("_pretrain","") for h in hs]
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for h, m in zip(hs, ms):
        e = json.loads(Path(h).read_text())
        ax.plot([x["epoch"] for x in e], [x["val_loss"] for x in e],
                marker="o", ms=3.5, lw=2, color=color_of(m, ms), label=m)
        b = int(np.argmin([x["val_loss"] for x in e]))
        ax.annotate(f"ppl {e[b]['val_ppl']:.2f}", (e[b]["epoch"], e[b]["val_loss"]),
                    textcoords="offset points", xytext=(5,-10), fontsize=8, color=INK2)
    ax.set_xlabel("epoch", fontsize=10, color=INK2)
    ax.set_ylabel("val loss  (MLM; gpt2: next-token)", fontsize=10, color=INK2)
    ax.set_title("Pretraining", color=INK, fontsize=11, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2); _style(ax)
    fig.tight_layout()
    sfx = _suffix(ms)
    _save(fig, f"pretrain_{sfx}", "pretrain_"); print(f"wrote pretrain_{sfx}")

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all","pretrain"):   fig_pretrain()
    if what in ("all","tasks"):      fig_tasks()
