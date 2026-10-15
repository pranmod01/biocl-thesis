#!/bin/bash
# Transfer pretraining sweep: 5 cfgs x 4 folds x 3 seeds = 60. Task i -> cfg i/12, fold (i%12)/3, seed i%3.
# Task 0 (pre_coarse f0 s0) was the calibration run (job 13220119). Runs from the transfer
# worktree so a branch switch in the main checkout can't change the code mid-sweep.
# ins048/060/091: NODE_FAIL / instant-fail during the first attempt (13235869, 10-07).
#SBATCH --job-name=transfer_sweep
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --exclude=ins058,ins082,ins048,ins060,ins091
#SBATCH --output=/insomnia001/home/pm3361/biocl-thesis-transfer/results/_jobs/transfer_sweep_%A_%a.log
set -uo pipefail
cd /insomnia001/home/pm3361/biocl-thesis-transfer
export PYTHONPATH=/insomnia001/home/pm3361/biocl-thesis-transfer
PY=./.venv/bin/python
CFGS=(pre_coarse pre_shuffle25 pre_shuffle50 pre_shuffle100 pre_fine)
i=$SLURM_ARRAY_TASK_ID
cfg=${CFGS[$((i / 12))]}; f=$(((i % 12) / 3)); s=$((i % 3))
echo "task $i: $cfg fold $f seed $s"
# resumable: a finished run leaves its checkpoint; never retrain it. Must LOAD, not just
# exist: a node failure right after torch.save left a full-size but truncated file (10-07).
ckpt=checkpoints/transfer/${cfg}_f${f}_s${s}.pt
if [ -f "$ckpt" ] && $PY -c "import torch, sys; torch.load(sys.argv[1], map_location='cpu', weights_only=False)" "$ckpt" 2>/dev/null; then
  echo "$ckpt exists and loads, skipping"; echo DONE; exit 0
fi
$PY -c "import torch; x = torch.randn(4096, 4096, device='cuda'); (x @ x).sum().item(); print('cuda ok', torch.cuda.get_device_name(0))" || { echo "NO CUDA, aborting"; exit 1; }
time $PY -m coarsecl.coarse.train_coarse --config configs/transfer/$cfg.yaml --split-fold $f --seed $s
rc=$?
echo "DONE rc=$rc"
exit $rc
