#!/bin/bash
#SBATCH -J ms4plus_SBP
#SBATCH -o /project/rhu/dpalfaro/results/ms4plus_SBP_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/ms4plus_SBP_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

CODE_DIR=/project/rhu/dpalfaro/code/HuPulse

# Use the shared ppgdata environment (adjust path if your conda is elsewhere)
# Option A: if using Yidan's environment directly
PYTHON=/home/yshen28/miniconda3/envs/ppgdata/bin/python
# Option B: if you installed your own copy via setup_env.sbatch
# PYTHON=/project/rhu/dpalfaro/conda/envs/ppgdata/bin/python

cd $CODE_DIR

echo "=== MS4Plus SBP Training ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

$PYTHON train.py --config configs/ms4_improved_SBP.yaml

echo "End: $(date)"
