#!/bin/bash
#SBATCH --job-name=novae
#SBATCH --output=/gpfs/workdir/blampeyq/.jobs_outputs/%j
#SBATCH --time=10:00:00
#SBATCH --partition=cpu_long
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8

module purge

module load anaconda3/2023.09-0/none-none

source activate novae

cd /gpfs/workdir/blampeyq/novae/scripts/experimental

python -u explore.py
