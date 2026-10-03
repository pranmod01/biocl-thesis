#!/bin/bash
# Item 15 on hippo (standalone 8x A100 box, no working SLURM): one run per GPU, detached
# so they survive the launching shell. Logs: results/_jobs/cl_combo_hippo_<cfg>.log
# Usage (from the repo root): bash results/_jobs/cl_combo_hippo.sh [first_gpu] [cfg ...]
# With no cfg names it launches the original Item 15 set.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=./.venv/bin/python
$PY -c "import torch; assert torch.cuda.is_available(), 'NO CUDA'" || exit 1
[ -f checkpoints/coarse_net.pt ] || echo "!!! checkpoints/coarse_net.pt missing: *_trained configs will fail"
gpu=${1:-0}
shift || true
cfgs=("$@")
[ ${#cfgs[@]} -eq 0 ] && cfgs=(i_replay_decay_oracle i_replay_aux_oracle i_replay_concat_trained i_replay_baseline_hippo)
for cfg in "${cfgs[@]}"; do
  log=results/_jobs/cl_combo_hippo_$cfg.log
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup bash -c \
    "echo \"start \$(date -Is) gpu=$gpu host=\$(hostname)\"; time $PY -m coarsecl.run --config configs/$cfg.yaml || echo '!!! $cfg FAILED'; echo \"end \$(date -Is)\"" \
    > "$log" 2>&1 < /dev/null &
  echo "launched $cfg on GPU $gpu (pid $!) -> $log"
  gpu=$((gpu + 1))
done
