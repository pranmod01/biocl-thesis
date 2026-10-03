#!/bin/bash
#SBATCH --job-name=cl_combo
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --exclude=ins082
#SBATCH --output=/insomnia001/home/pm3361/biocl-thesis/results/_jobs/cl_combo_%j.log
set -uo pipefail
cd /insomnia001/home/pm3361/biocl-thesis
PY=./.venv/bin/python
$PY -c "import torch; assert torch.cuda.is_available(), 'NO CUDA'; print('cuda:', torch.cuda.get_device_name(0))" || exit 1
# The trained-source run needs the coarse-net checkpoint; fail that run loudly, not late.
[ -f checkpoints/coarse_net.pt ] || echo "!!! checkpoints/coarse_net.pt missing: i_replay_concat_trained will fail"
# Item 15: the mechanisms never tried on top of replay, all deployable (no superclass
# label at test time). Single seed: read each against the replay-2000 twins
# (i_replay_cal2000 0.310 / i_replay_baseline 0.379), not against one number.
for cfg in i_replay_decay_oracle i_replay_aux_oracle i_replay_concat_trained; do
  echo "########## RUN $cfg ##########"
  time $PY -m coarsecl.run --config configs/$cfg.yaml || echo "!!! $cfg FAILED"
done
echo "=== DONE ==="
