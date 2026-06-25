#!/bin/bash
#SBATCH --job-name=transv2     # ????
#SBATCH --output=transv3_noise_sbp%j.out   # ?????
#SBATCH --error=transv3_noise_sbp%j.out    # ?????
#SBATCH -N 1                          # ??1???
#SBATCH -n 1                          # ??4???
#SBATCH --gres=gpu:2               # ??1?GPU volta:
#SBATCH -p gpu                  # ??gpu??
#SBATCH --mem=300G                     # ????
#SBATCH --time=4-00:00:00             # ?????? 1 ?

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export CUDA_HOME=/share/apps/cuda/11.3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# ??Python????
srun -u python Model_Training/Model_Training_transv2.py \
