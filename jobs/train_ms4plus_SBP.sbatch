#!/bin/bash
#SBATCH --job-name=MS4film_SBP
#SBATCH --output=MS4film_SBP_%j.out
#SBATCH --error=MS4film_SBP_%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:2
#SBATCH -p gpu
#SBATCH --mem=300G
#SBATCH --time=7-00:00:00

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export CUDA_HOME=/share/apps/cuda/11.3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

CODE_DIR=/project/rhu/PulseBP/Pulse/PulseDB_multi_full

srun -u python $CODE_DIR/scripts/train_ms4plus_carya.py --bp SBP --variant film
