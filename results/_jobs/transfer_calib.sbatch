#!/bin/bash
#SBATCH --job-name=transfer_calib
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --exclude=ins082
#SBATCH --output=/insomnia001/home/pm3361/biocl-thesis/results/_jobs/transfer_pre_coarse_f0_s0_%j.log
set -uo pipefail
cd /insomnia001/home/pm3361/biocl-thesis
export PYTHONPATH=/insomnia001/home/pm3361/biocl-thesis
PY=./.venv/bin/python
# device: auto would silently fall back to CPU on a bad node; fail instead.
$PY -c "import torch, sys; ok = torch.cuda.is_available(); print('cuda', ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)" || { echo "NO CUDA, aborting"; exit 1; }
time $PY -m coarsecl.coarse.train_coarse --config configs/transfer/pre_coarse.yaml --split-fold 0 --seed 0
echo DONE
