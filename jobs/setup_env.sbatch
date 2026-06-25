#!/bin/bash
#SBATCH --job-name=setup_ppgdata
#SBATCH --output=setup_ppgdata_%j.out
#SBATCH --error=setup_ppgdata_%j.out
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -p batch
#SBATCH --time=2:00:00

# Carya's shared conda base (from `conda env list` base entry)
CONDA_BASE=/project/dsi/apps/easybuild/software/Miniforge3/24.1.2-0
source $CONDA_BASE/etc/profile.d/conda.sh

ENV_PREFIX=/project/rhu/dpalfaro/conda/envs/ppgdata
YML=$HOME/ppgdata_environment.yml

echo "Creating ppgdata env at $ENV_PREFIX"
conda env create -f $YML --prefix $ENV_PREFIX --force

echo "Done. Python at: $ENV_PREFIX/bin/python"
$ENV_PREFIX/bin/python --version
