#!/bin/bash
#SBATCH -J explore_pulsedb
#SBATCH -o /project/rhu/dpalfaro/results/explore_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/explore_%j.err
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

CODE_DIR=/project/rhu/dpalfaro/code/HuPulse
PYTHON=/home/yshen28/miniconda3/envs/ppgdata/bin/python

cd $CODE_DIR
$PYTHON scripts/explore_data.py --data_root /project/rhu/PulseBP
