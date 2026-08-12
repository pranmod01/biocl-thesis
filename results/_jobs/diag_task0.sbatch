#!/bin/bash
#SBATCH --job-name=diag_task0
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --exclude=ins082
#SBATCH --output=/insomnia001/home/pm3361/biocl-thesis/results/_jobs/diag_task0_%j.log
set -uo pipefail
cd /insomnia001/home/pm3361/biocl-thesis
export PYTHONPATH=/insomnia001/home/pm3361/biocl-thesis
time ./.venv/bin/python results/_jobs/diag_task0.py
echo DONE
