#!/bin/bash
#SBATCH -p pi_abodner
#SBATCH --job-name=MLD_diagnostic
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00-16:00:00
#SBATCH -o /home/codycruz/LLC_ocean_emulator/high_res_diagnostics/MLD/logs/%x-%j.out
#SBATCH -e /home/codycruz/LLC_ocean_emulator/high_res_diagnostics/MLD/logs/%x-%j.out
#SBATCH --hint=nomultithread

set -euo pipefail

start=$(date +%s)

# load module
module load miniforge/24.3.0-0

# set location of script
location=/home/codycruz/LLC_ocean_emulator/high_res_diagnostics/MLD

# Job type flag
job_type="grid_1month" # grid_1month or ts

echo "Job:$job_type"

# Memory profiling flag
scalene=False # True or False
export SCALENE_PROFILE=$scalene

# activate virtual environment
source /home/codycruz/LLC_ocean_emulator/high_res_diagnostics/.venv/bin/activate

# -----------------------------------------------------------------------------
# Fast data-first defaults (override any of these with sbatch --export or env)
# -----------------------------------------------------------------------------
export MLD_MAKE_FIGURES=${MLD_MAKE_FIGURES:-False}
export MLD_TEMPORAL_AVG=${MLD_TEMPORAL_AVG:-1MS}     # 1MS, 1D, 1H, or none
export MLD_TILE_WIDTH=${MLD_TILE_WIDTH:-0.25}

export MLD_FACE=${MLD_FACE:-1}
export MLD_I0=${MLD_I0:-2880}
export MLD_I1=${MLD_I1:-3600}
export MLD_J0=${MLD_J0:-720}
export MLD_J1=${MLD_J1:-1440}
export MLD_T0=${MLD_T0:-9216}
export MLD_T1=${MLD_T1:-9960}

export MLD_N_WORKERS=${MLD_N_WORKERS:-2}
export MLD_WRITE_TIME_CHUNK=${MLD_WRITE_TIME_CHUNK:-24}
export MLD_WRITE_TILE_CHUNK=${MLD_WRITE_TILE_CHUNK:-64}
export MLD_MAX_FIGURES=${MLD_MAX_FIGURES:-200}
export MLD_DATA_DIR=${MLD_DATA_DIR:-/orcd/data/abodner/002/cody/MLD_diagnostic_data}
export MLD_SOURCE_ZARR=${MLD_SOURCE_ZARR:-/orcd/data/abodner/003/LLC4320/LLC4320}

echo "======== calculate MLD of a spatiotemporal subset of the LLC4320 dataset ========"
echo "scalene: $scalene"

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "CPUS=$SLURM_CPUS_PER_TASK"
echo "MEM=$SLURM_MEM_PER_NODE"

echo "MLD_MAKE_FIGURES=$MLD_MAKE_FIGURES"
echo "MLD_TEMPORAL_AVG=$MLD_TEMPORAL_AVG"
echo "MLD_TILE_WIDTH=$MLD_TILE_WIDTH"
echo "MLD_FACE=$MLD_FACE"
echo "MLD_I0=$MLD_I0, MLD_I1=$MLD_I1"
echo "MLD_J0=$MLD_J0, MLD_J1=$MLD_J1"
echo "MLD_T0=$MLD_T0, MLD_T1=$MLD_T1"
echo "MLD_N_WORKERS=$MLD_N_WORKERS"
echo "MLD_WRITE_TIME_CHUNK=$MLD_WRITE_TIME_CHUNK"
echo "MLD_WRITE_TILE_CHUNK=$MLD_WRITE_TILE_CHUNK"
echo "MLD_DATA_DIR=$MLD_DATA_DIR"
echo "MLD_SOURCE_ZARR=$MLD_SOURCE_ZARR"

if [ "$scalene" = "True" ]; then
    mkdir -p "$location/scalene/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
    JSON_OUT="$location/scalene/${SLURM_JOB_NAME}-${SLURM_JOB_ID}/json.json"

    uv run python -m scalene run \
        --cpu-only \
        -o "$JSON_OUT" \
        "$location/MLD_${job_type}.py"

    cd "$location/scalene/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
    uv run python -m scalene view --html-file "$JSON_OUT"
else
    uv run "$location/MLD_${job_type}.py"
fi

end=$(date +%s)
runtime=$((end-start))
printf "Total runtime: %02d:%02d:%02d\n" \
  $((runtime/3600)) $((runtime%3600/60)) $((runtime%60))

echo "======== job complete ========"
