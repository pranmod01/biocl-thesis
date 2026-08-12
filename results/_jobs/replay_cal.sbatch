#!/bin/bash
#SBATCH --job-name=replay_cal
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --exclude=ins082
#SBATCH --output=/insomnia001/home/pm3361/biocl-thesis/results/_jobs/replay_cal_%j.log
set -uo pipefail
cd /insomnia001/home/pm3361/biocl-thesis
PY=./.venv/bin/python
$PY -c "import torch; assert torch.cuda.is_available(), 'NO CUDA'; print('cuda:', torch.cuda.get_device_name(0))" || exit 1
for cfg in i_replay_cal500 i_replay_cal2000 i_replay_cal5000; do
  echo "########## RUN $cfg ##########"
  time $PY -m coarsecl.run --config configs/$cfg.yaml || echo "!!! $cfg FAILED"
done
echo "=== DONE ==="
