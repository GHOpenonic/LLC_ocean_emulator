#!/bin/bash
#SBATCH -p pi_abodner
#SBATCH --job-name=MLD_global_exp:1_rerun
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=400GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --time=02-04:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.out
#SBATCH --hint=nomultithread

start=$(date +%s)

# load module
module load miniforge/24.3.0-0

# set location of script
location=/home/codycruz/LLC_ocean_emulator/high_res_diagnostics/MLD/global

echo "Job:$job_type"

# Memory profiling flag =================================================
scalene=False # True or False
export SCALENE_PROFILE=True

# activate virtual environment
source /home/codycruz/LLC_ocean_emulator/high_res_diagnostics/.venv/bin/activate

echo "======== calculate MLD of a spatiotemporal subset of the LLC4320 dataset, GLOBAL TEST ========"
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
    uv run "$location/MLD_global.py"
fi


end=$(date +%s)
runtime=$((end-start))
printf "Total runtime: %02d:%02d:%02d\n" \
  $((runtime/3600)) $((runtime%3600/60)) $((runtime%60))

echo "======== job complete ========"