#!/bin/bash
#SBATCH --job-name=diag_ceiling
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --exclude=ins082
#SBATCH --output=/insomnia001/home/pm3361/biocl-thesis/results/_jobs/diag_ceiling_%j.log
set -uo pipefail
cd /insomnia001/home/pm3361/biocl-thesis
export PYTHONPATH=/insomnia001/home/pm3361/biocl-thesis
./.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'NO CUDA'; print('cuda:', torch.cuda.get_device_name(0))" || exit 1
time ./.venv/bin/python results/_jobs/diag_ceiling.py
echo DONE
