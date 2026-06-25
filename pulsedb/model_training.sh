#!/bin/bash
#SBATCH -J bootstrap_resnet50_10fold
#SBATCH -o bootstrap_resnet50_10fold.o%j
#SBATCH --mail-user=namathe4@cougarnet.uh.edu
#SBATCH --mail-type=BEGIN,FAIL,END

#SBATCH --ntasks-per-node=8 -N 1
#SBATCH -t 96:0:0
#SBATCH --mem=64G

#SBATCH --gres=gpu:volta:1

source ~/.bashrc

conda activate /project/zouridakis/gzlab2/pulsedb_env

# matlab -r /project/zouridakis/gzlab2/PulseDB/Generate_Subsets.m
# env > /project/zouridakis/gzlab2/env-sbatch.txt
export PYTHONUNBUFFERED=1

# python3 /project/zouridakis/gzlab2/PulseDB/Model_Training/train_xresnet1d18.py
# python3 /project/zouridakis/gzlab2/PulseDB/pulsedb_yidan_multi/Model_Training/Model_Training.py
python3 /project/zouridakis/gzlab2/PulseDB/Model_Training/model_training_bootstrap.py