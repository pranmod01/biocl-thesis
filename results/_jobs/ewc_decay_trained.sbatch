#!/bin/bash
#SBATCH --job-name=ewc_decay_trained
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --exclude=ins082
#SBATCH --output=/insomnia001/home/pm3361/biocl-thesis/results/_jobs/ewc_decay_trained_%j.log
set -uo pipefail
cd /insomnia001/home/pm3361/biocl-thesis
PY=./.venv/bin/python
$PY -c "import torch; assert torch.cuda.is_available(), 'NO CUDA'; print('cuda:', torch.cuda.get_device_name(0))" || exit 1
[ -f checkpoints/coarse_net.pt ] || { echo "!!! checkpoints/coarse_net.pt missing"; exit 1; }
cfg=i_replay_ewc_decay_trained
echo "########## RUN $cfg ##########"
time $PY -m coarsecl.run --config configs/$cfg.yaml || echo "!!! $cfg FAILED"
echo "=== DONE ==="
