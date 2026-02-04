#!/bin/bash
#SBATCH -p pi_abodner
#SBATCH --job-name=MLD_2_Kuroshio_zarr
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=360GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=01-01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.out
#SBATCH --hint=nomultithread

start=$(date +%s)

# load module
module load miniforge/24.3.0-0

# set location of script
location=/home/codycruz/LLC_ocean_emulator/high_res_diagnostics/MLD

# Job type Flag ===========================================================
# select which script is run:\
#job_type="grid_1" # calculates MLD per pixel column -> averages spatially -> averages temporally
job_type="grid_2" # averages spatially -> calculates MLD per spatial tile -> averages temporally
# job_type="grid_3" # averages spatially -> averages temporally -> calculates MLD per spatial tile per month
#job_type="ts" # calculates MLD time series in a spatial box

echo "Job:$job_type"

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
        "$location/MLD_${job_type}.py"

    # produce an html of the JSON profile
    cd "$location/scalene/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
    uv run python -m scalene view --html-file "$JSON_OUT"

else
    # run the script without memory profiling
    uv run "$location/MLD_${job_type}.py"
fi


end=$(date +%s)
runtime=$((end-start))
printf "Total runtime: %02d:%02d:%02d\n" \
  $((runtime/3600)) $((runtime%3600/60)) $((runtime%60))

echo "======== job complete ========"