#!/bin/bash
#SBATCH -J dataset_analysis
#SBATCH -o dataset_analysis.o%j
#SBATCH --mail-user=namathe4@cougarnet.uh.edu
#SBATCH --mail-type=END

#SBATCH --ntasks-per-node=10 -N 1
#SBATCH -t 24:0:0
#SBATCH --mem-per-cpu=32GB

#SBATCH -p cpu

# ml torchvision
source ~/.bashrc

# conda activate /project/zouridakis/gzlab2/pulsedb_env

module add matlab
cd /project/zouridakis/gzlab2/PulseDB

matlab -r Generate_Subsets
