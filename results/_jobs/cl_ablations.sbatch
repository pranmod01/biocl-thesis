#!/bin/bash
#SBATCH --job-name=cl_abl
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --exclude=ins082
#SBATCH --output=/insomnia001/home/pm3361/biocl-thesis/results/_jobs/cl_abl_%j.log
set -uo pipefail
cd /insomnia001/home/pm3361/biocl-thesis
PY=./.venv/bin/python
$PY -c "import torch; assert torch.cuda.is_available(), 'NO CUDA'; print('cuda:', torch.cuda.get_device_name(0))" || exit 1
# Baselines first: they define the new floor every coarse condition is read against.
# EWC runs carry a per-step penalty over all params, so they are the slow ones.
for cfg in i_replay_baseline i_ewc_baseline i_replay_ewc_baseline \
           i_replay_gate_oracle i_replay_concat_oracle i_replay_gate_trained; do
  echo "########## RUN $cfg ##########"
  time $PY -m coarsecl.run --config configs/$cfg.yaml || echo "!!! $cfg FAILED"
done
echo "=== DONE ==="
