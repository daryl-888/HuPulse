#!/bin/bash
#SBATCH -J datasetcheck
#SBATCH -o datasetcheck.o%j

#SBATCH --ntasks-per-node=12 -N 1
#SBATCH -t 3:0:0
#SBATCH --mem-per-cpu=32GB

#SBATCH -p cpu

# ml torchvision
source ~/.bashrc

conda activate /project/zouridakis/gzlab2/pulsedb_env

export PYTHONUNBUFFERED=1

# python3 /project/zouridakis/gzlab2/PulseDB/Model_Training/revised_pi.py
# python3 /project/zouridakis/gzlab2/PulseDB/Model_Training/bland_altman.py

python3 /project/zouridakis/gzlab2/dataset_subjects.py