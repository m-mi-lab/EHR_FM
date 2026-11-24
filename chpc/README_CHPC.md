# mamba pretrain on CHPC PE (OnDemand Jupyter, H200, 72 h sessions)

CHPC only allows DUO-interactive logins, so the launch is typed by hand;
everything else is scripted. Set `LAB=user@lab-host` and `LABROOT=<repo root there>`. Metrics/logs are synced to the lab box so
progress is visible there (and in MLflow) without touching the PE again.

## once: unattended access from redwood to the lab box

    ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 -q <<< y >/dev/null 2>&1
    ssh-copy-id $LAB                                  # lab-box password, once

## once: get the bundle (needs ~7 GB of quota)

    ROOT=<that dir>/ICLR; mkdir -p $ROOT && cd $ROOT
    scp $LAB:$LABROOT/chpc_code.tar $LAB:$LABROOT/chpc_bundle.tar .
    tar xf chpc_code.tar && tar xf chpc_bundle.tar && rm chpc_code.tar chpc_bundle.tar

`chpc_code.tar` = code, configs, launchers, this dir. `chpc_bundle.tar` (3.2 GB) =
the final EIC tokenization stage (all the training reads), its vocabulary,
labels, and the two raw MIMIC tables the demographic prefix needs (`raw/hosp/patients.csv`, `admissions.csv`).
The old `EHR_FM/data/mimic-iv-meds/shard_events` on CHPC is the pre-fix
corpus (missing 400 M chartevents rows) -- do not use it.

## once: environment (Jupyter terminal, inside the Apptainer container)

    cd $ROOT && bash chpc/setup_chpc.sh

## every session: launch / resume

    cd $ROOT && bash chpc/launch.sh mamba && tail -f logs/pre_mamba.log

Other archs can share the H200 the same way (`bash chpc/launch.sh gpt2s`);
each takes one 32,768-token micro-batch per step (~25-30 GB).

Geometry is mamba-130m (24 blocks, d=768; 96.6 M with this vocab), ctx
2048, bf16, same ETHOS windows / plateau-LR / patience-5 recipe as the
other models; one micro-batch of
32,768-token micro-batch per step. `--resume 1` continues from
`data/models/mamba_pretrain/last.pt` (saved every epoch) when a session
is restarted. A sync loop copies the log, `history.json` and the local
MLflow store to the lab box (`logs/chpc/`) every 10 minutes.

## when it converges: bring back

    data/models/mamba_pretrain/best.pt   (the sync loop does not copy it)

    scp $ROOT/data/models/mamba_pretrain/best.pt $LAB:$LABROOT/data/models/mamba_pretrain/

then `run_experiments.sh mamba <gpu>` on the lab box (finetunes,
missingness sweeps, figures need the held-out set and the other
models' results, which live there).
