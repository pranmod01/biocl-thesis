#!/bin/bash
#SBATCH --job-name=decay2
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --exclude=ins082
#SBATCH --output=/insomnia001/home/pm3361/biocl-thesis/results/_jobs/decay2_%j.log
set -uo pipefail
cd /insomnia001/home/pm3361/biocl-thesis
PY=./.venv/bin/python
$PY -c "import torch; assert torch.cuda.is_available(), 'NO CUDA'; print('cuda:', torch.cuda.get_device_name(0))" || exit 1
for cfg in g_decay_oracle_w05 g_decay_oracle_w20 g_decay_oracle_w50 h_taskmask_baseline h_decay_taskmask; do
  echo "########## RUN $cfg ##########"
  time $PY -m coarsecl.run --config configs/$cfg.yaml || echo "!!! $cfg FAILED"
done
echo "=== DONE ==="
