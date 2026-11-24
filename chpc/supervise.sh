#!/bin/bash
# CHPC job supervisor: start once per OnDemand session, then leave it alone.  (rev 2026-09-27b: resets restart counters)
#     cd ~/ICLR && nohup bash chpc/supervise.sh > logs/supervise.log 2>&1 &
# Every cycle it
#   * pulls chpc_code.tar + chpc/jobs.txt from the lab box (so the job list and code can be
#     changed from the lab box without touching this terminal),
#   * launches every `pretrain <arch> <gpu>` job that is not running and not converged,
#     one at a time, waiting for each to finish loading and for cgroup memory headroom
#     (five simultaneous checkpoint loads is what OOM-killed mambal on 2026-09-24),
#   * relaunches jobs that die (max RESTART_MAX per arch per session),
#   * runs `sweep <arch> <gpu> <task...>` jobs whose finetuned checkpoints are final on the
#     lab box (its <arch>_ft/<task>/result.json exists) or whose pretrain CONVERGED here
#     (chpc/sweep.sh then finetunes first), and whose results are not done yet,
#   * copies its own log to the lab box.
set -uo pipefail
# ROOT survives the self-re-exec below (after exec, $0 is the /tmp copy)
[ -n "${ICLR_SUP_ROOT:-}" ] || ICLR_SUP_ROOT=$(cd "$(dirname "$0")/.." && pwd)
export ICLR_SUP_ROOT; ROOT=$ICLR_SUP_ROOT; cd "$ROOT" || exit 1
LAB=${LAB:?set LAB=user@lab-host}; LABROOT=${LABROOT:?set LABROOT=<repo root on the lab box>}
CYCLE=${CYCLE:-180}; MEM_HEADROOM_GB=${MEM_HEADROOM_GB:-20}; LOAD_TIMEOUT=${LOAD_TIMEOUT:-1500}
RESTART_MAX=${RESTART_MAX:-5}; RESTART_GAP=${RESTART_GAP:-300}; MEM_LIMIT_GB=${MEM_LIMIT_GB:-64}
export JOBS=${JOBS:-chpc/jobs.txt} SUPERVISE_LOG=${SUPERVISE_LOG:-logs/supervise.log} MLFLOW_SUFFIX=${MLFLOW_SUFFIX:-}   # per-session: a second session uses its own list/log/store
mkdir -p logs data/results/missingness
declare -A restarts lastfail

# run from a private copy so re-extracting the code tar never rewrites the running script
if [ "${SUPERVISE_COPY:-}" != "$0" ]; then
  cp "$ROOT/chpc/supervise.sh" /tmp/supervise_run_$$.sh; SUPERVISE_COPY=/tmp/supervise_run_$$.sh exec bash /tmp/supervise_run_$$.sh
fi
exec 8>/tmp/iclr_supervise2${MLFLOW_SUFFIX:-}.lock; flock -n 8 || { echo "$(date "+%m-%d %H:%M:%S") another supervisor holds /tmp/iclr_supervise2${MLFLOW_SUFFIX:-}.lock, exiting"; exit 0; }
SELF_MD5=$(md5sum chpc/supervise.sh | cut -c1-32)
log() { echo "$(date '+%m-%d %H:%M:%S') $*"; }

mem_avail() {   # prints "<free GB> <source>" under the job's memory cap: its cgroup (v1/v2) if visible, else Slurm request minus total RSS; "" if unknown
  local lim="" use="" cg src=""
  cg=$(awk -F: '$2 ~ /(^|,)memory(,|$)/ {print $3; exit}' /proc/self/cgroup 2>/dev/null)
  if [ -n "$cg" ] && [ -r "/sys/fs/cgroup/memory$cg/memory.limit_in_bytes" ]; then
    lim=$(cat "/sys/fs/cgroup/memory$cg/memory.limit_in_bytes"); use=$(cat "/sys/fs/cgroup/memory$cg/memory.usage_in_bytes"); src="cgroup-v1:$cg"
  else
    cg=$(awk -F: '$1=="0" {print $3; exit}' /proc/self/cgroup 2>/dev/null)
    if [ -n "$cg" ] && [ -r "/sys/fs/cgroup$cg/memory.max" ]; then lim=$(cat "/sys/fs/cgroup$cg/memory.max"); use=$(cat "/sys/fs/cgroup$cg/memory.current"); src="cgroup-v2:$cg"; fi
  fi
  if [ -z "$lim" ] || [ "$lim" = max ] || [ "${#lim}" -gt 15 ]; then     # no usable cgroup view (container shows an unlimited root)
    lim=$(( $(mem_limit_gb) * 1073741824 )); src="$(mem_limit_gb)GB-limit-minus-anon"
    use=$(pss_bytes)
  fi
  echo "$(( (lim - use) / 1073741824 )) $src"
}
mem_limit_gb() {   # the job's memory cap in GB: Slurm's per-node or per-CPU request (visible in the container), else `memlimit` from jobs.txt
  if [ -n "${SLURM_MEM_PER_NODE:-}" ]; then echo $(( SLURM_MEM_PER_NODE / 1024 ))
  elif [ -n "${SLURM_MEM_PER_CPU:-}" ]; then echo $(( SLURM_MEM_PER_CPU * ${SLURM_CPUS_ON_NODE:-1} / 1024 ))
  else echo "$MEM_LIMIT_GB"; fi
}
pss_bytes() {   # memory actually charged to our processes (shared pages counted once); RSS sum as a fallback
  local p s=0
  for p in $(pgrep -u "${USER:-$(id -un)}"); do   # proportional (Pss) anonymous + shmem mappings only: memory-mapped file pages are reclaimable page cache
    v=$(awk '/^[0-9a-f]+-[0-9a-f]+ /{a=($6=="" || $6 ~ /^\[(heap|stack|anon)/ || $6 ~ /^\/dev\/(shm|zero)/ || $6 ~ /SYSV/); next} /^Pss:/{if (a) s+=$2} END{print s+0}' /proc/$p/smaps 2>/dev/null); s=$(( s + ${v:-0} )); done
  [ "$s" -gt 0 ] || s=$(ps -u "${USER:-$(id -un)}" -o rss= | awk '{s+=$1} END{print s+0}')
  echo $(( s * 1024 ))
}
wait_headroom() {   # block (at most HEADROOM_TIMEOUT s) until the job has $1 GB free; returns 1 on timeout so the caller skips this cycle
  local need=${1:-$MEM_HEADROOM_GB} a t0=$SECONDS
  while :; do a=$(mem_avail); a=${a%% *}; [ -z "$a" ] && return 0; [ "$a" -ge "$need" ] && return 0
    [ $((SECONDS - t0)) -ge "${HEADROOM_TIMEOUT:-1200}" ] && { log "  no memory headroom (${a} GB free < ${need} GB), trying again next cycle"; return 1; }
    log "  waiting for memory headroom (${a} GB free < ${need} GB)"; sleep 60; done
}
need_gb() { case "$1" in gpt2l|mambal|mistrall|bigbird) echo 12;; *) echo 3;; esac; }   # private-memory transient of a launch (checkpoints are memory-mapped)
pull_control() {   # refresh code + job list from the lab box; re-exec if this script changed
  scp -q -o BatchMode=yes -o ConnectTimeout=15 "$LAB:$LABROOT/chpc_code.tar" /tmp/chpc_code_new.tar 2>/dev/null || return
  if ! cmp -s /tmp/chpc_code_new.tar chpc_code.tar 2>/dev/null; then
    mv /tmp/chpc_code_new.tar chpc_code.tar && tar xf chpc_code.tar && log "code updated from lab box"
  fi
  local m; m=$(md5sum chpc/supervise.sh | cut -c1-32)   # checked every cycle: another supervisor sharing ~/ICLR may have extracted the tar
  if [ "$m" != "$SELF_MD5" ]; then log "supervisor changed, re-exec"; cp chpc/supervise.sh /tmp/supervise_run_$$.sh
    SUPERVISE_COPY=/tmp/supervise_run_$$.sh exec bash /tmp/supervise_run_$$.sh; fi
}
pre_running() { pgrep -f "[p]retrain.py --model $1 " >/dev/null; }
main_pids() {   # main trainer processes of an arch (parent is the run_pretrain.sh wrapper); loader workers are excluded
  local p; for p in $(pgrep -f "[p]retrain.py --model $1 "); do ps -o args= -p "$(ps -o ppid= -p "$p" | tr -d " ")" 2>/dev/null | grep -q run_pretrain.sh && echo "$p"; done; }
dedup() {   # if an arch has more than one main process (a launch race), keep the oldest
  local arch=$1 extra; extra=$(main_pids "$arch" | sort -n | tail -n +2)
  [ -n "$extra" ] && { log "$arch: duplicate trainer(s) $(echo $extra) killed, keeping the oldest"; kill $extra; }; }
pre_converged() {   # CONVERGED within the latest launch block of the arch's log
  local f=logs/pre_$1.log h; [ -f "$f" ] || return 1
  h=$(grep -anE "^=== pretrain" "$f" | tail -1 | cut -d: -f1); [ -n "$h" ] || return 1
  tail -n +"$h" "$f" | grep -aq "^CONVERGED"
}
wait_loaded() {   # block until launch #$2 of $1 shows RESUMED (or vocab= for a fresh run)
  local arch=$1 nhdr=$2 f=logs/pre_$1.log key=vocab= t0=$SECONDS h
  [ -f "data/models/${arch}_pretrain/last.pt" ] && key=RESUMED
  while [ $((SECONDS - t0)) -lt "$LOAD_TIMEOUT" ]; do
    if [ "$(grep -ac "^=== pretrain" "$f" 2>/dev/null)" -ge "$nhdr" ]; then   # the new launch's header is in
      h=$(grep -anE "^=== pretrain" "$f" | tail -1 | cut -d: -f1)
      tail -n +"$h" "$f" | grep -aq "^$key" && { log "  $arch loaded ($key)"; return; }
      pre_running "$arch" || { log "  $arch exited during load"; return; }
    fi
    sleep 20
  done; log "  $arch load wait timed out after ${LOAD_TIMEOUT}s, continuing"
}
fits_session() {   # false (and one log line) when the session's memory cap cannot hold this job's load at all
  local what=$1 need=$2 lim; lim=$(mem_limit_gb)
  [ $((need + 4)) -le "$lim" ] && return 0
  [ -z "${skipped[$what]:-}" ] && { log "$what: skipped, needs ~$((need + 4)) GB of memory and this session has ${lim} GB"; skipped[$what]=1; }
  return 1
}
declare -A skipped sweep_launches
launch_pretrain() {   # pretrain <arch> <gpu> [wait:<marker>] [unless:<marker>]: markers are logs/<marker>.done
  local arch=$1 gpu=$2; shift 2; local w
  for w in "$@"; do case "$w" in wait:*) [ -f "logs/${w#wait:}.done" ] || return;; unless:*) [ -f "logs/${w#unless:}.done" ] && return;; esac; done
  local n=${restarts[$arch]:-0}; local nhdr
  fits_session "$arch" "$(need_gb "$arch")" || return
  if [ "$n" -ge "$RESTART_MAX" ]; then
    [ "$n" -eq "$RESTART_MAX" ] && { log "$arch: giving up after $RESTART_MAX launches this session (will retry hourly)"; restarts[$arch]=$((n+1)); }
    [ $((SECONDS - ${lastfail[$arch]:-0})) -ge 3600 ] || return   # a neighbour's transient GPU-memory peak is not a broken run: try again once an hour
    log "$arch: retrying an hour after giving up"; restarts[$arch]=0; n=0; fi
  if [ "$n" -gt 0 ] && [ $((SECONDS - ${lastfail[$arch]:-0})) -lt "$RESTART_GAP" ]; then return; fi
  wait_headroom "$(need_gb "$arch")" || return
  local nh; nh=$(grep -ac "^=== pretrain" "logs/pre_$arch.log" 2>/dev/null); nhdr=$(( ${nh:-0} + 1 ))   # grep -c prints 0 and exits 1 on no match, so a fallback echo would print a second 0
  log "launch $arch on GPU $gpu (attempt $((n+1)))"; bash chpc/launch.sh "$arch" "$gpu" 2>&1 | sed 's/^/  /'
  restarts[$arch]=$((n+1)); lastfail[$arch]=$SECONDS
  wait_loaded "$arch" "$nhdr"
}
sweep_running() { pgrep -f "[s]weep.sh $1 " >/dev/null; }
declare -A dcount lastcpu
kill_stuck() {   # D state + zero CPU growth for 40 cycles (~2 h) = hung for good: kill so the pipeline moves on
  local pid st cpu args
  while read -r pid st cpu args; do
    if [[ $st == D* ]] && [ "${lastcpu[$pid]:-}" = "$cpu" ]; then dcount[$pid]=$(( ${dcount[$pid]:-0} + 1 )); else dcount[$pid]=0; fi
    lastcpu[$pid]=$cpu
    if [ "${dcount[$pid]}" -ge 40 ]; then   # ~2 h: a process waiting on slow NFS page reads is alive (D state, no CPU, no majflt growth) and finishes
      log "stuck: pid $pid in D state with no CPU for ${dcount[$pid]} cycles, killed: $(cut -c1-90 <<<"$args")"; kill -9 "$pid" 2>/dev/null; dcount[$pid]=0
    fi
  done < <(ps -eo pid,stat,times,args | awk '/[p]retrain.py|[f]inetune.py|[r]un_missingness.py/ {print $1, $2, $3, $4, $5, $6, $7, $8, $9}')
}
ensure_ckpt() {   # pull <arch>_ft/<task>/best.pt if the lab box marks that finetune finished
  local arch=$1 task=$2 d
  d=data/models/${arch}_ft/$task
  [ -f "$d/best.pt" ] && return 0
  if [ -f "data/models/${arch}_pretrain/best.pt" ]; then   # pretrained here (CONVERGED in our log) or pulled from the lab box: sweep.sh finetunes
    { pre_converged "$arch" || [ -f "data/models/${arch}_pretrain/.from_lab" ]; } && return 0; fi
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$LAB" "test -f $LABROOT/$d/result.json" 2>/dev/null || return 1
  mkdir -p "$d"; log "  pulling $d/best.pt from the lab box"
  scp -q -o BatchMode=yes "$LAB:$LABROOT/$d/best.pt" "$d/best.pt" && return 0; rm -f "$d/best.pt"; return 1
}
redo_task() {   # redo <arch> <task>: once, drop a partial finetune and its results so the sweep line redoes the task from the pretrain checkpoint
  local arch=$1 task=$2; local mark="logs/redo_${arch}_${task}.done"   # two statements: a local list expands every word before assigning any
  [ -f "$mark" ] && return
  pkill -f "[s]weep.sh $arch " 2>/dev/null && log "redo $arch/$task: stopped the running sweep"
  sleep 3; pkill -9 -f "(run_missingness|finetune).py --model $arch --task $task" 2>/dev/null; sleep 2
  rm -rf "data/models/${arch}_ft/$task" "data/results/missingness/$arch/$task.event.json"
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$LAB" "rm -rf $LABROOT/data/models/${arch}_ft/$task $LABROOT/data/results/missingness/$arch/$task.event.json" 2>/dev/null
  date > "$mark"; log "redo $arch/$task: partial finetune and results removed here and on the lab box; the sweep line redoes it"
}
stop_arch() {   # stop <arch> [now|epoch] [tag]: kill this arch's pretrain here, once per tag. epoch = wait for the next last.pt save (nothing lost);
  local arch=$1 mode=${2:-now} tag=${3:-}   # tag to:<x> = handover: waits for logs/supervisor_<x>.alive (the receiving session) to be fresh; any other tag is a marker name
  local to=""; case "$tag" in to:*) to=${tag#to:}; tag=$to;; esac
  local mark="logs/stop_${arch}${tag:+_$tag}.done"; local req="logs/stop_${arch}${tag:+_$tag}.req"
  [ -f "$mark" ] && return
  if [ -n "$to" ] && [ -z "$(find logs -maxdepth 1 -name "supervisor_${to}.alive" -mmin -15 2>/dev/null)" ]; then return; fi
  [ -f "$req" ] || { date +%s > "$req"; log "stop $arch ($mode${to:+, handover to $to}): requested"; }
  if [ "$mode" = epoch ] && pre_running "$arch"; then
    local ck="data/models/${arch}_pretrain/last.pt"
    { [ -f "$ck" ] && [ ! -f "$ck.tmp" ] && [ "$(stat -c %Y "$ck")" -gt $(( $(cat "$req") + 60 )) ]; } || return   # no save since the request yet
  fi
  pkill -f "[r]un_pretrain.sh [0-9]+ .*$arch$" 2>/dev/null; pkill -f "[p]retrain.py --model $arch " 2>/dev/null; sleep 5; pkill -9 -f "[p]retrain.py --model $arch " 2>/dev/null
  date > "$mark"; log "stop $arch: pretrain stopped here after its last save (resume elsewhere from last.pt)"
}
reset_arch() {   # reset <arch> [tag]: once per tag, kill this arch's pretrain and wipe its checkpoints so the pretrain line restarts it from scratch
  local arch=$1 tag=${2:-}; local mark="logs/reset_${arch}${tag:+_$tag}.done"
  [ -f "$mark" ] && return
  pkill -f "[r]un_pretrain.sh [0-9]+ .*$arch$" 2>/dev/null; pkill -f "[p]retrain.py --model $arch " 2>/dev/null; sleep 5; pkill -9 -f "[p]retrain.py --model $arch " 2>/dev/null
  rm -f "data/models/${arch}_pretrain/last.pt" "data/models/${arch}_pretrain/best.pt" "data/models/${arch}_pretrain/history.json"
  restarts[$arch]=0; date > "$mark"; log "reset $arch: pretrain killed and checkpoints removed; the pretrain line restarts it from scratch"
}
rewind_arch() {   # rewind <arch> <epoch> [tag]: once per tag, stop the arch, keep best.pt, cut history.json to epochs <= N, drop last.pt; the pretrain line resumes it from best.pt
  local arch=$1 ep=$2 tag=${3:-}
  local mark="logs/rewind_${arch}_${ep}${tag:+_$tag}.done"; local d="data/models/${arch}_pretrain"
  [ -f "$mark" ] && return
  pkill -f "[r]un_pretrain.sh [0-9]+ .*$arch$" 2>/dev/null; pkill -f "[p]retrain.py --model $arch " 2>/dev/null; sleep 5; pkill -9 -f "[p]retrain.py --model $arch " 2>/dev/null
  [ -f "$d/history.json" ] && "$ROOT/.venv/bin/python" -c "import json,sys; p,n=sys.argv[1],int(sys.argv[2]); h=json.load(open(p)); json.dump([e for e in h if e['epoch']<=n], open(p,'w'), indent=1)" "$d/history.json" "$ep"
  rm -f "$d/last.pt"; echo "=== pretrain $arch rewound to epoch $ep  $(date) ===" >> "logs/pre_$arch.log"
  restarts[$arch]=0; date > "$mark"; log "rewind $arch to epoch $ep: last.pt removed, history.json cut, best.pt kept; the pretrain line resumes it from best.pt"
}
pull_pretrain() {   # pull_pretrain <arch>: once, fetch the lab box's converged pretrain checkpoint so the arch's sweep line finetunes it here
  local arch=$1; local d="data/models/${arch}_pretrain"
  [ -f "$d/.from_lab" ] && return
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$LAB" "grep -aq 'PRETRAIN DONE' $LABROOT/logs/pre_$arch.log" 2>/dev/null || return
  mkdir -p "$d"; log "  pulling $d/best.pt from the lab box (lab-pretrained arch)"
  if scp -q -o BatchMode=yes "$LAB:$LABROOT/$d/best.pt" "$d/best.pt" && scp -q -o BatchMode=yes "$LAB:$LABROOT/$d/history.json" "$d/history.json"; then
    date > "$d/.from_lab"; log "pull_pretrain $arch: checkpoint pulled; its sweep line finetunes here"
  else rm -f "$d/best.pt"; log "pull_pretrain $arch: scp failed, retrying next cycle"; fi
}
rm_path() {   # rm <path>: once per path, delete a redundant checkpoint file/dir under data/models or data/results and log the size freed
  local path=$1; local mark="logs/rm_$(echo "$path" | tr '/' '_').done"
  [ -f "$mark" ] && return
  case "$path" in data/models/*|data/results/*) ;; *) log "rm $path: refused (outside data/models, data/results)"; date > "$mark"; return;; esac
  if [ -e "$path" ]; then local sz; sz=$(du -sh "$path" 2>/dev/null | cut -f1); rm -rf "$path" && log "rm $path: removed ($sz)"; else log "rm $path: not present"; fi
  date > "$mark"
}
run_sweeps() {
  local arch=$1 gpu=$2; shift 2; local pending=()
  sweep_running "$arch" && return
  for t in "$@"; do   # a task is done when its result is here or on the lab box (the lab-swept models have no copy here)
    [ -f "data/results/missingness/$arch/$t.event.json" ] && continue
    ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$LAB" "test -f $LABROOT/data/results/missingness/$arch/$t.event.json" 2>/dev/null
    case $? in 1) ;; *) continue;; esac   # 0: the lab box has it; 255: lab box unreachable, ask again next cycle; only a clear no is pending
    ensure_ckpt "$arch" "$t" && pending+=("$t")
  done
  [ ${#pending[@]} -eq 0 ] && return
  if [ "${sweep_launches[$arch]:-0}" -ge 4 ]; then
    [ "${sweep_launches[$arch]}" -eq 4 ] && { log "sweep $arch: giving up after 4 launches this session (${pending[*]} still pending)"; sweep_launches[$arch]=5; }
    return; fi
  fits_session "sweep-$arch" 4 || return
  wait_headroom 4 || return; log "sweep $arch on GPU $gpu: ${pending[*]}"; sweep_launches[$arch]=$(( ${sweep_launches[$arch]:-0} + 1 ))
  nohup bash chpc/sweep.sh "$arch" "$gpu" "${pending[@]}" >> "logs/sweep_$arch.log" 2>&1 8>&- &
  sleep 180   # let it import torch and load before anything else starts
}
run_sweepx() {   # sweepx <arch> <gpu> <families> <tasks...>: extra missingness families into <task>.visits_event.json
  local arch=$1 gpu=$2 fams=$3; shift 3; local pending=() t key="x$arch"
  sweep_running "$arch" && return
  [ "$(pgrep -fc '[s]weep.sh ')" -ge "${SWEEP_MAX:-5}" ] && return   # at most SWEEP_MAX sweep.sh processes visible on this node (pgrep sees the other sessions on a shared node, so 5 leaves room for ~3 visit runs beside two regular sweeps)
  for t in "$@"; do   # only tasks whose FINISHED finetune is on this disk (ensure_ckpt pulls lab-box finetunes; CHPC-finetuned ones appear via the regular chain)
    [ -f "data/results/missingness/$arch/$t.visits_event.json" ] && continue
    local d="data/models/${arch}_ft/$t"
    if [ ! -f "$d/best.pt" ]; then   # not on this disk (lab-trained, or removed in the quota cleanup): pull it if the lab box marks the finetune finished
      if ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$LAB" "test -f $LABROOT/$d/result.json" 2>/dev/null; then
        mkdir -p "$d"; scp -q -o BatchMode=yes "$LAB:$LABROOT/$d/best.pt" "$d/best.pt" </dev/null && log "  pulled $d/best.pt from the lab box" || rm -f "$d/best.pt"
      fi
    fi
    [ -f "$d/best.pt" ] && [ -f "$d/result.json" ] && pending+=("$t")   # result.json = the finetune FINISHED (a killed one leaves best.pt alone)
  done
  [ ${#pending[@]} -eq 0 ] && return
  if [ "${sweep_launches[$key]:-0}" -ge 4 ]; then
    [ "${sweep_launches[$key]}" -eq 4 ] && { log "sweepx $arch: giving up after 4 launches this session (${pending[*]} still pending)"; sweep_launches[$key]=5; }
    return; fi
  fits_session "sweepx-$arch" 4 || return
  wait_headroom 4 || return; log "sweepx $arch on GPU $gpu ($fams): ${pending[*]}"; sweep_launches[$key]=$(( ${sweep_launches[$key]:-0} + 1 ))
  FAMILIES="$fams" SUFFIX=visits_event MIN_VISITS="${MIN_VISITS:-3}" nohup bash chpc/sweep.sh "$arch" "$gpu" "${pending[@]}" >> "logs/sweepx_$arch.log" 2>&1 8>&- &
  sleep 180
}
run_sweepe() {   # sweepe <arch> <gpu> <tasks...>: the six token families under EVENT removal (removal before tokenisation) into <task>.event.json
  local arch=$1 gpu=$2 fams=A1,A3,B1,C1,C2,C3; shift 2; local pending=() t key="e$arch"
  sweep_running "$arch" && return
  [ "$(pgrep -fc '[s]weep.sh ')" -ge "${SWEEP_MAX:-5}" ] && return   # at most SWEEP_MAX sweep.sh processes visible on this node (pgrep sees the other sessions on a shared node, so 5 leaves room for ~3 visit runs beside two regular sweeps)
  for t in "$@"; do   # only tasks whose FINISHED finetune is on this disk (ensure_ckpt pulls lab-box finetunes; CHPC-finetuned ones appear via the regular chain)
    [ -f "data/results/missingness/$arch/$t.event.json" ] && continue
    local d="data/models/${arch}_ft/$t"
    if [ ! -f "$d/best.pt" ]; then   # not on this disk (lab-trained, or removed in the quota cleanup): pull it if the lab box marks the finetune finished
      if ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$LAB" "test -f $LABROOT/$d/result.json" 2>/dev/null; then
        mkdir -p "$d"; scp -q -o BatchMode=yes "$LAB:$LABROOT/$d/best.pt" "$d/best.pt" </dev/null && log "  pulled $d/best.pt from the lab box" || rm -f "$d/best.pt"
      fi
    fi
    [ -f "$d/best.pt" ] && [ -f "$d/result.json" ] && pending+=("$t")   # result.json = the finetune FINISHED (a killed one leaves best.pt alone)
  done
  [ ${#pending[@]} -eq 0 ] && return
  if [ "${sweep_launches[$key]:-0}" -ge 4 ]; then
    [ "${sweep_launches[$key]}" -eq 4 ] && { log "sweepe $arch: giving up after 4 launches this session (${pending[*]} still pending)"; sweep_launches[$key]=5; }
    return; fi
  fits_session "sweepe-$arch" 4 || return
  wait_headroom 4 || return; log "sweepe $arch on GPU $gpu (event removal): ${pending[*]}"; sweep_launches[$key]=$(( ${sweep_launches[$key]:-0} + 1 ))
  FAMILIES="$fams" SUFFIX=event nohup bash chpc/sweep.sh "$arch" "$gpu" "${pending[@]}" >> "logs/sweepx_$arch.log" 2>&1 8>&- &
  sleep 180
}

resync_results() {   # ship every result JSON the lab box does not have at the same size (cheap: a few KB each)
  local f rel want have
  want=$(find data/results -name "*.json" -printf "%p %s\n" 2>/dev/null; find -H data/models -maxdepth 3 -name result.json -printf "%p %s\n" 2>/dev/null)
  [ -n "$want" ] || return 0
  have=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "$LAB" "cd $LABROOT && find data/results -name '*.json' -printf '%p %s\n' 2>/dev/null; cd $LABROOT && find -H data/models -maxdepth 3 -name result.json -printf '%p %s\n' 2>/dev/null" 2>/dev/null) || return 0
  while read -r f sz; do
    grep -qxF "$f $sz" <<<"$have" && continue
    ssh -o BatchMode=yes -o ConnectTimeout=15 "$LAB" "mkdir -p $LABROOT/$(dirname "$f")" 2>/dev/null && \
      scp -q -o BatchMode=yes "$f" "$LAB:$LABROOT/$f" 2>/dev/null && log "  resynced $f to the lab box"
  done <<<"$want"
}

log "supervisor up in $ROOT (cycle ${CYCLE}s, headroom ${MEM_HEADROOM_GB} GB, mem avail now: $(mem_avail) GB)"
while :; do
  pull_control; date > "logs/supervisor${MLFLOW_SUFFIX:-}.alive"   # freshness marker other sessions key handovers on
  while read -r kind arch gpu rest <&3; do   # pass 1: settings and duplicate cleanup, never blocks (fd 3: an ssh inside the loop must not drain the job list)
    case "$kind" in memlimit) MEM_LIMIT_GB=$arch;; pretrain) dedup "$arch";; redo) redo_task "$arch" "$gpu";; stop) stop_arch "$arch" "$gpu" $rest;; reset) reset_arch "$arch" "$gpu";; rewind) rewind_arch "$arch" "$gpu" $rest;; pull_pretrain) pull_pretrain "$arch";; rm) rm_path "$arch";; esac
  done 3< <(grep -vE '^\s*(#|$)' "$JOBS" 2>/dev/null)
  kill_stuck
  while read -r kind arch gpu rest <&3; do   # pass 2: launches (each may wait for headroom / loading); job list on fd 3, see pass 1
    case "$kind" in
      pretrain) pre_running "$arch" || pre_converged "$arch" || launch_pretrain "$arch" "$gpu" $rest;;
      sweepx) run_sweepx "$arch" "$gpu" $rest;;
      sweepe) run_sweepe "$arch" "$gpu" $rest;;
      sweep)    run_sweeps "$arch" "$gpu" $rest;;
    esac
  done 3< <(grep -vE '^\s*(#|$)' "$JOBS" 2>/dev/null)
  for f in data/results/missingness/*/*.json; do   # sweeps started before 2026-10-08 still write the old MNAR ids
    [ -f "$f" ] && grep -q '"family": "C4"' "$f" && "$ROOT/.venv/bin/python" scripts/renumber_families.py "$f" >/dev/null 2>&1 && log "  renumbered families in $f"
  done
  resync_results   # results whose scp failed while the lab box was unreachable
  for t in $(find data/models/ -maxdepth 2 -name "*.pt.tmp" -mmin +60 2>/dev/null); do rm -f "$t" && log "removed stale partial checkpoint $t"; done
  if ! pgrep -f "[c]hpc_sync_loop${MLFLOW_SUFFIX:-} " >/dev/null; then   # no sync loop of our own: ship pretrain logs + histories too
    scp -q -o BatchMode=yes -o ConnectTimeout=15 logs/pre_*.log "$LAB:$LABROOT/logs/chpc/" 2>/dev/null
    for d in data/models/*_pretrain; do [ -f "$d/history.json" ] && scp -q -o BatchMode=yes -o ConnectTimeout=15 "$d/history.json" "$LAB:$LABROOT/$d/" 2>/dev/null; done
  fi
  db="mlflow_chpc${MLFLOW_SUFFIX:-}.db"   # no sync loop of our own (second session on a shared node): ship the store ourselves
  if [ -f "$db" ] && ! pgrep -f "[c]hpc_sync_loop${MLFLOW_SUFFIX:-} " >/dev/null && [ "$(stat -c %Y "$db")" != "${db_shipped:-}" ]; then
    scp -q -o BatchMode=yes -o ConnectTimeout=15 "$db" "$LAB:$LABROOT/logs/chpc/" 2>/dev/null && db_shipped=$(stat -c %Y "$db"); fi
  { date; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
    ps -eo pid,etime,rss,maj_flt,pcpu,stat,wchan:22,args --sort=-rss | grep -E "[p]retrain.py|[f]inetune.py|[r]un_missingness.py|[s]weep.sh" | cut -c1-150
    for d in $(ps -eo pid,stat,args | awk '/[p]retrain.py|[f]inetune.py|[r]un_missingness.py/ && $2 ~ /^D/ {print $1}'); do
      echo "D-state $d: syscall=[$(cat /proc/$d/syscall 2>/dev/null | cut -c1-60)] wchan=[$(cat /proc/$d/wchan 2>/dev/null)] cmd=$(tr '\0' ' ' < /proc/$d/cmdline | cut -c1-80)"
      ls -l /proc/$d/fd 2>/dev/null | awk '{print $NF}' | grep -E "^/" | grep -vE "/dev/pts|/dev/null|\.so|python" | sort -u | tail -n 12 | sed 's/^/    fd: /'
    done
    echo "mem avail: $(mem_avail) GB"
    ps -eo pid,etime,args | grep -E "[s]upervise" | grep -v "grep" | cut -c1-120 | sed 's/^/supervisor: /'
    echo "cache: $(ls data/eic/cache/ 2>/dev/null | tr '\n' ' ') $(ls data/eic/cache/*/.building 2>/dev/null)"
    echo "disk: home $(df -h "$ROOT" 2>/dev/null | awk 'NR==2{print $3"/"$2" used, "$4" free"}') | models -> $(readlink -f data/models) $(df -h "$(readlink -f data/models)" 2>/dev/null | awk 'NR==2{print $3"/"$2" used, "$4" free"}')"
    command -v mydiskquota >/dev/null && mydiskquota 2>/dev/null | grep -vE "^\s*$" | head -12 | sed 's/^/quota: /'
    du -sh data/models/*/ 2>/dev/null | sort -h | tail -10 | sed 's/^/size: /'; du -sh "$ROOT"/{logs,data/eic,mlflow_chpc*.db,*.tar} 2>/dev/null | sort -h | tail -6 | sed 's/^/size: /'
    for f in logs/sweep_*.log; do [ -f "$f" ] && { echo "== $f"; grep -aE "^=== |^--- |building|waiting|LEAKCHECK|^wrote|FAILED|Killed|Traceback|rror" "$f" | tail -n 4 | cut -c1-150; tail -n 2 "$f" | cut -c1-150; }; done; } > logs/chpc_status${MLFLOW_SUFFIX:-}.txt 2>&1   # what is running here, for the lab box
  scp -q -o BatchMode=yes -o ConnectTimeout=15 "$SUPERVISE_LOG" logs/chpc_status${MLFLOW_SUFFIX:-}.txt "$LAB:$LABROOT/logs/chpc/" 2>/dev/null
  sleep "$CYCLE"
done
