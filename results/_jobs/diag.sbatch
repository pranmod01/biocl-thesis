#!/bin/bash
#SBATCH --job-name=coarsecl_diag
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --exclude=ins082
#SBATCH --output=/insomnia001/home/pm3361/biocl-thesis/results/_jobs/diag_%j.log
set -uo pipefail
cd /insomnia001/home/pm3361/biocl-thesis
./.venv/bin/python -c "import torch; print('cuda:', torch.cuda.is_available())"
time ./.venv/bin/python -m coarsecl.run --config results/_probe_cfgs/diag_base_60ep.yaml
echo DONE
