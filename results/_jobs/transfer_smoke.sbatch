#!/bin/bash
#SBATCH --job-name=transfer_smoke
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --exclude=ins082
#SBATCH --output=/insomnia001/home/pm3361/biocl-thesis/results/_jobs/transfer_smoke_%j.log
set -uo pipefail
cd /insomnia001/home/pm3361/biocl-thesis
export PYTHONPATH=/insomnia001/home/pm3361/biocl-thesis
PY=./.venv/bin/python
$PY -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "=== split fold 0, shuffle 0.5, 5 ep ==="
$PY -m coarsecl.coarse.train_coarse --config results/_jobs/transfer_smoke5_split.yaml --split-fold 0
echo "=== plain 20-way, 5 ep ==="
$PY -m coarsecl.coarse.train_coarse --config results/_jobs/transfer_smoke5_plain.yaml
echo "=== test_cl_methods ==="
$PY tests/test_cl_methods.py
echo DONE
