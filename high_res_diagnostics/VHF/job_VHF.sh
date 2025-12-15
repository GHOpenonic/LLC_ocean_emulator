#!/bin/bash
#SBATCH -p pi_abodner
#SBATCH --job-name=VHF
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=900GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03-12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.out


# load module
module load miniforge/24.3.0-0

# activate uv environment for ocean_emulator
uv sync --dev

echo "======== calculate VHF of a spatiotemporal subset of the LLC4320 dataset ========"

uv run /home/codycruz/high_res_diagnostics/VHF/VHF.py

echo "======== job complete ========"