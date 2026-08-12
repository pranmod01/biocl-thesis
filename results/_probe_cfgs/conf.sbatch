#!/bin/bash
#SBATCH --job-name=coarsecl_conf
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/insomnia001/home/pm3361/biocl-thesis/results/_probe_logs/conf_%j.log

set -uo pipefail
cd /insomnia001/home/pm3361/biocl-thesis
PY=./.venv/bin/python
CFG=/insomnia001/home/pm3361/biocl-thesis/results/_probe_cfgs

echo "=== node: $(hostname) ==="
$PY -c "import torch; print('cuda:', torch.cuda.is_available())"

for cfg in conf_base_5x20 conf_oracle_5x20 conf_base_10x10_20ep conf_oracle_10x10_20ep; do
  echo "=== $cfg ==="
  time $PY -m coarsecl.run --config $CFG/$cfg.yaml || echo "!!! $cfg FAILED"
done
echo "=== DONE ==="
