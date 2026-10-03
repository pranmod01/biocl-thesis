#!/bin/bash
# Item 12: multi-seed the deployable coarse leads against the replay-2000 control.
#
# Four arms x seeds 1,2,3, paired: `--seed` varies run.seed only (init, shuffling,
# replay sampling), so data.order_seed stays 0 and every arm sees the SAME task
# stream. The existing seed-0 runs are the 4th sample of each arm, so each arm ends
# at n=4 (the control has 4 seed-0 samples of its own on top of that).
#
# 12 jobs over 8 GPUs, round-robin: GPU g runs job g then job g+8, sequentially.
# ~15 min/run => ~30 min wall clock. Detached, survives the launching shell.
# Logs: results/_jobs/seed_sweep_<cfg>_s<seed>.log
# Usage (from the repo root): bash results/_jobs/seed_sweep_hippo.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=./.venv/bin/python
$PY -c "import torch; assert torch.cuda.is_available(), 'NO CUDA'" || exit 1
[ -f checkpoints/coarse_net.pt ] || { echo "!!! checkpoints/coarse_net.pt missing (gate_trained needs it)"; exit 1; }

ARMS=(i_replay_baseline_hippo i_replay_decay_oracle i_replay_aux_oracle i_replay_gate_trained)
SEEDS=(1 2 3)
NGPU=8

jobs=()
for s in "${SEEDS[@]}"; do for a in "${ARMS[@]}"; do jobs+=("$a:$s"); done; done

for ((g = 0; g < NGPU; g++)); do
  slot=()
  for ((j = g; j < ${#jobs[@]}; j += NGPU)); do slot+=("${jobs[$j]}"); done
  [ ${#slot[@]} -eq 0 ] && continue
  cmd=""
  for job in "${slot[@]}"; do
    cfg=${job%:*}; seed=${job#*:}
    log=results/_jobs/seed_sweep_${cfg}_s${seed}.log
    cmd+="{ echo \"start \$(date -Is) gpu=$g host=\$(hostname) cfg=$cfg seed=$seed\"; "
    cmd+="time $PY -m coarsecl.run --config configs/$cfg.yaml --seed $seed "
    cmd+="|| echo '!!! ${cfg}_s${seed} FAILED'; echo \"end \$(date -Is)\"; } > $log 2>&1; "
  done
  CUDA_VISIBLE_DEVICES=$g setsid nohup bash -c "$cmd" > /dev/null 2>&1 < /dev/null &
  echo "GPU $g <- ${slot[*]} (pid $!)"
done
echo
echo "watch:  tail -n 3 results/_jobs/seed_sweep_*.log"
