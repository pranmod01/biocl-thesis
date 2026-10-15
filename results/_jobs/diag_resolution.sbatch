#!/bin/bash
#SBATCH --job-name=diag_resolution
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --exclude=ins082,ins090
#SBATCH --array=0-5
#SBATCH --output=/insomnia001/home/pm3361/biocl-thesis/results/_jobs/diag_resolution_%A_%a.log
set -uo pipefail
cd /insomnia001/home/pm3361/biocl-thesis
PY=./.venv/bin/python
$PY -c "import torch; assert torch.cuda.is_available(), 'NO CUDA'; print('cuda:', torch.cuda.get_device_name(0))" || exit 1
# Premise check for the resolution curriculum: does low res keep superclass info better
# than fine info? Joint training, {fine, coarse} labels x {8, 16, 32} px, seed 0.
# Analyze with: python results/_jobs/diag_resolution_analyze.py
LABELS=(fine fine fine coarse coarse coarse)
RES=(32 16 8 32 16 8)
i=$SLURM_ARRAY_TASK_ID
time $PY results/_jobs/diag_resolution.py --labels ${LABELS[$i]} --res ${RES[$i]} --seed 0 \
  || echo "!!! ${LABELS[$i]} r${RES[$i]} FAILED"
echo "=== DONE ==="
