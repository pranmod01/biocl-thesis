#!/bin/bash
#SBATCH --job-name=transfer_eval_dev
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --exclude=ins082,ins048,ins060,ins091
#SBATCH --output=/insomnia001/home/pm3361/biocl-thesis-transfer/results/_jobs/transfer_eval_dev_%j.log
set -uo pipefail
cd /insomnia001/home/pm3361/biocl-thesis-transfer
export PYTHONPATH=/insomnia001/home/pm3361/biocl-thesis-transfer
time ./.venv/bin/python -m coarsecl.transfer.evaluate --config results/_jobs/transfer_eval_dev.yaml
echo DONE
