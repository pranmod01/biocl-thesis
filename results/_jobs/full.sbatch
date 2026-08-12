#!/bin/bash
#SBATCH --job-name=coarsecl_full
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/insomnia001/home/pm3361/biocl-thesis/results/_jobs/full_%j.log

set -uo pipefail
cd /insomnia001/home/pm3361/biocl-thesis
PY=./.venv/bin/python

echo "=== node: $(hostname) ==="
$PY -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

echo "########## PRETRAIN coarse net (condition d prerequisite) ##########"
time $PY -m coarsecl.coarse.train_coarse --config configs/d_trained.yaml || { echo "!!! coarse pretrain FAILED"; exit 1; }

for cfg in a_baseline b_oracle c_soft d_trained; do
  echo "########## RUN $cfg ##########"
  time $PY -m coarsecl.run --config configs/$cfg.yaml || echo "!!! $cfg FAILED"
done

echo "=== DONE ==="
