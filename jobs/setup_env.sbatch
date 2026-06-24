#!/bin/bash
# Create the ppgdata conda environment from the shared environment.yml.
# Run once: sbatch jobs/setup_env.sbatch
# NOTE: This job does NOT need a GPU.
#SBATCH -J setup_ppgdata_env
#SBATCH -o /project/rhu/dpalfaro/results/setup_env_%j.out
#SBATCH -e /project/rhu/dpalfaro/results/setup_env_%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dpalfaro@cougarnet.uh.edu

CODE_DIR=/project/rhu/dpalfaro/code/HuPulse

# If you have a personal conda installation, adjust the path:
CONDA_BASE=/project/rhu/dpalfaro/conda
source $CONDA_BASE/etc/profile.d/conda.sh

# Create env from the shared yml (name will be 'ppgdata' as per the yml)
# --prefix installs into your project space to avoid home-dir quota issues
conda env create \
    -f $CODE_DIR/ppgdata_environment.yml \
    --prefix $CONDA_BASE/envs/ppgdata \
    --force

echo "Environment ready at: $CONDA_BASE/envs/ppgdata"
