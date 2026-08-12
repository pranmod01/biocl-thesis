#!/bin/bash
#SBATCH --job-name=coarsecl_probe
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/insomnia001/home/pm3361/biocl-thesis/results/_probe_logs/probe_%j.log

set -uo pipefail

cd /insomnia001/home/pm3361/biocl-thesis
PY=./.venv/bin/python
CFG=/insomnia001/home/pm3361/biocl-thesis/results/_probe_cfgs

echo "=== node: $(hostname) ==="
$PY -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

for cfg in probe_base_lr01 probe_oracle_concat_lr01 probe_oracle_aux_lr1 probe_oracle_aux_lr01; do
  echo "=== $cfg ==="
  time $PY -m coarsecl.run --config $CFG/$cfg.yaml || echo "!!! $cfg FAILED"
done

echo "=== DONE ==="
