#!/bin/bash
# Item 22 on hippo: launch configs round-robin over a GPU list, all concurrently
# (a replay-2000 run uses ~1.8 GB and well under a full A100, so 2-3 share a GPU).
# Detached, survives the launching shell. Logs: results/_jobs/noise_sweep_<cfg>.log
# Usage (from the repo root): bash results/_jobs/noise_sweep_hippo.sh "5 6 7" cfg [cfg ...]
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=./.venv/bin/python
$PY -c "import torch; assert torch.cuda.is_available(), 'NO CUDA'" || exit 1
read -ra gpus <<< "$1"; shift
i=0
for cfg in "$@"; do
  g=${gpus[$((i % ${#gpus[@]}))]}
  log=results/_jobs/noise_sweep_$cfg.log
  CUDA_VISIBLE_DEVICES=$g setsid nohup bash -c \
    "echo \"start \$(date -Is) gpu=$g host=\$(hostname)\"; time $PY -m coarsecl.run --config configs/$cfg.yaml || echo '!!! $cfg FAILED'; echo \"end \$(date -Is)\"" \
    > "$log" 2>&1 < /dev/null &
  echo "GPU $g <- $cfg (pid $!)"
  i=$((i + 1))
done
