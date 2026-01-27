#!/bin/bash
#SBATCH -p pi_abodner
#SBATCH --job-name=MLD_
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=250GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00-08:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.out
#SBATCH --hint=nomultithread

# load module
module load miniforge/24.3.0-0

# set location of script
location=/home/codycruz/LLC_ocean_emulator/high_res_diagnostics/MLD

# Memory profiling flag =================================================
scalene=False # True or False
export SCALENE_PROFILE=True

# activate virtual environment
source /home/codycruz/LLC_ocean_emulator/high_res_diagnostics/.venv/bin/activate

echo "======== calculate MLD of a spatiotemporal subset of the LLC4320 dataset ========"
echo "scalene: $scalene"

# some diagnostics
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "CPUS=$SLURM_CPUS_PER_TASK"
echo "MEM=$SLURM_MEM_PER_NODE"

if [ "$scalene" = "True" ]; then
    mkdir -p "$location/scalene/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
    JSON_OUT="$location/scalene/${SLURM_JOB_NAME}-${SLURM_JOB_ID}/json.json"

    # run scalene and produce JSON profile
    uv run python -m scalene run \
        --cpu-only \
        -o "$JSON_OUT" \
        "$location/MLD_ts.py"

    # produce an html of the JSON profile
    cd "$location/scalene/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
    uv run python -m scalene view --html-file "$JSON_OUT"

else
    # run the script without memory profiling
    uv run "$location/MLD_per_pixel.py"
fi

echo "======== job complete ========"