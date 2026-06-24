#!/bin/bash
#SBATCH -J ms4plus_DBP
#SBATCH -o /project/rhu/dpalfaro/results/ms4plus_DBP_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/ms4plus_DBP_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=ada:1
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

CODE_DIR=/project/rhu/dpalfaro/code/HuPulse

PYTHON=/home/yshen28/miniconda3/envs/ppgdata/bin/python
# PYTHON=/project/rhu/dpalfaro/conda/envs/ppgdata/bin/python

cd $CODE_DIR

echo "=== MS4Plus DBP Training ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

$PYTHON train.py --config configs/ms4_improved_DBP.yaml

echo "End: $(date)"
