#!/bin/bash
#SBATCH -J ms4plus_SBP
#SBATCH -o /project/rhu/dpalfaro/results/ms4plus_SBP_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/ms4plus_SBP_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

CODE_DIR=/project/rhu/dpalfaro/code/HuPulse

# Use Yidan's shared ppgdata environment (Python 3.8 + PyTorch 1.10.1+cu113 + mat73)
PYTHON=/home/yshen28/miniconda3/envs/ppgdata/bin/python
# If you installed your own copy via setup_env.sbatch, use instead:
# PYTHON=/project/rhu/dpalfaro/conda/envs/ppgdata/bin/python

mkdir -p /project/rhu/dpalfaro/results

cd $CODE_DIR

echo "=== MS4Plus SBP — 10-fold CV ==="
echo "Node : $(hostname)"
echo "GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

$PYTHON scripts/train_ms4plus_carya.py --bp SBP

echo "End: $(date)"
