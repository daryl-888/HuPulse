#!/bin/bash
#SBATCH --job-name=setup_ppgdata
#SBATCH --output=setup_ppgdata_%j.out
#SBATCH --error=setup_ppgdata_%j.out
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -p batch
#SBATCH --time=2:00:00

set -e

CONDA_BASE=/project/dsi/apps/easybuild/software/Miniforge3/24.1.2-0
source $CONDA_BASE/etc/profile.d/conda.sh

ENV_PREFIX=/project/rhu/dpalfaro/conda/envs/ppgdata
YML=$HOME/ppgdata_environment.yml

# Remove stale partial install if it exists
if [ -d "$ENV_PREFIX" ]; then
    echo "Removing existing partial env at $ENV_PREFIX"
    conda env remove --prefix $ENV_PREFIX -y
fi

echo "Creating ppgdata env at $ENV_PREFIX"
conda env create -f $YML --prefix $ENV_PREFIX

echo "Done. Python at: $ENV_PREFIX/bin/python"
$ENV_PREFIX/bin/python --version
