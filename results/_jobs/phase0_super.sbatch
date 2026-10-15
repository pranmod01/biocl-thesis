#!/bin/bash
#SBATCH --job-name=phase0_super
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --exclude=ins082
#SBATCH --array=0-3
#SBATCH --output=/insomnia001/home/pm3361/biocl-thesis/results/_jobs/phase0_super_%A_%a.log
set -uo pipefail
cd /insomnia001/home/pm3361/biocl-thesis
PY=./.venv/bin/python
$PY -c "import torch; assert torch.cuda.is_available(), 'NO CUDA'; print('cuda:', torch.cuda.get_device_name(0))" || exit 1
# Phase 0: rerun seed 0 of the conditions with NO coarse signal at test time, now that
# eval also scores superclass accuracy. Question: when old-task fine accuracy drops, is
# the superclass kept (leaf forgotten) or lost too? `_super` suffix keeps the original
# result dirs intact; fine-scale numbers should match them up to GPU nondeterminism.
CFGS=(a_baseline i_replay_baseline_hippo i_replay_decay_oracle i_replay_aux_oracle)
cfg=${CFGS[$SLURM_ARRAY_TASK_ID]}
echo "########## RUN $cfg ##########"
time $PY -m coarsecl.run --config configs/$cfg.yaml --name ${cfg}_super || echo "!!! $cfg FAILED"
echo "=== DONE ==="
