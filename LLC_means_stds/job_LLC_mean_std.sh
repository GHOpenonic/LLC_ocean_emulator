#!/bin/bash
#SBATCH -p pi_abodner
#SBATCH --job-name=LLC_mean_std_calc
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=680GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=03-00:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.out


# load module
module load miniforge/24.3.0-0

echo "======== calculate means and stds of the LLC4320 dataset ========"

uv run /home/codycruz/LLC_means_stds/LLC_mean_std.py

echo "======== job complete ========"