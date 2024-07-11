#!/bin/bash 
#SBATCH --time=00:00:00
#SBATCH --mem=0G
#SBATCH --account pmg
#SBATCH --nodelist=m003

module load mamba
mamba init
source /burg/home/*your username*/.bashrc
mamba activate *dir*/mambaforge/*env name*

