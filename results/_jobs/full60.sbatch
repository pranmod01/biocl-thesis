#!/bin/bash
#SBATCH --job-name=coarsecl_full60
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --exclude=ins082
#SBATCH --output=/insomnia001/home/pm3361/biocl-thesis/results/_jobs/full60_%j.log

set -uo pipefail
cd /insomnia001/home/pm3361/biocl-thesis
PY=./.venv/bin/python

echo "=== node: $(hostname) ==="
$PY -c "import torch; assert torch.cuda.is_available(), 'NO CUDA'; print('cuda:', torch.cuda.get_device_name(0))" || exit 1

echo "########## PRETRAIN coarse net (60ep) ##########"
time $PY -m coarsecl.coarse.train_coarse --config configs/d_trained.yaml || { echo "!!! coarse pretrain FAILED"; exit 1; }

# a/b/c/d = the four original conditions (now 60ep); e_gate_* = variant B; b_oracle_withhold = variant A
for cfg in a_baseline b_oracle c_soft d_trained b_oracle_withhold e_gate_oracle e_gate_soft e_gate_trained; do
  echo "########## RUN $cfg ##########"
  time $PY -m coarsecl.run --config configs/$cfg.yaml || echo "!!! $cfg FAILED"
done

echo "=== DONE ==="
