#!/bin/bash
# One condition of the val-split eval: sbatch transfer_eval.sbatch <condition>
#SBATCH --job-name=transfer_eval
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --exclude=ins058,ins082,ins048,ins060,ins091
#SBATCH --output=/insomnia001/home/pm3361/biocl-thesis-transfer/results/_jobs/transfer_eval_%x_%j.log
set -uo pipefail
cd /insomnia001/home/pm3361/biocl-thesis-transfer
export PYTHONPATH=/insomnia001/home/pm3361/biocl-thesis-transfer
PY=./.venv/bin/python
$PY -c "import torch; x = torch.randn(4096, 4096, device='cuda'); (x @ x).sum().item(); print('cuda ok', torch.cuda.get_device_name(0))" || { echo "NO CUDA, aborting"; exit 1; }
time $PY -m coarsecl.transfer.evaluate --config configs/transfer/eval_val.yaml --only "$1"
rc=$?
echo "DONE rc=$rc"
exit $rc
