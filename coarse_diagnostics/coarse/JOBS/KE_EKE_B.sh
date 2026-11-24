#!/bin/bash
#SBATCH -p pi_abodner
#SBATCH --job-name=2026-11-12:coarse:KE_EKE_B_ALLTIME-1month-384vs640
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=00-06:00:00
#SBATCH -o /home/codycruz/LLC_ocean_emulator/coarse_diagnostics/coarse/logs/%x-%j.out
#SBATCH -e /home/codycruz/LLC_ocean_emulator/coarse_diagnostics/coarse/logs/%x-%j.out

set -euo pipefail

# ============== PARAMETERS ==============
# One NAME=PATH per emulator; override with EMULATORS="a=/p/a.zarr b=/p/b.zarr".
DEFAULT_EMULATORS=(
  "w384_3hr_eager_ckpt36=/orcd/scratch/seedfund/001/llc_emulator/inference/coarse//w384-ckpt36-3hr-1month-t12/predictions.zarr"
  "w640_3hr_eager_ckpt35=/orcd/scratch/seedfund/001/llc_emulator/inference/coarse/w640-ckpt35-3hr-1month-t12/predictions.zarr"
)


if [[ -n "${EMULATORS:-}" ]]; then read -r -a EMULATOR_LIST <<< "$EMULATORS"; else EMULATOR_LIST=("${DEFAULT_EMULATORS[@]}"); fi
VARS="${VARS:-KE EKE B}"  # any of: KE EKE B
MODE="${MODE:-map time_series}"  # map and/or time_series
FACE="${FACE:-all}"  # 0-12 for one face, all for the globe
X_EXTENT="${X_EXTENT:-0 270}"  # start stop, per face
Y_EXTENT="${Y_EXTENT:-0 270}"  # start stop, per face
DEPTH="${DEPTH:-0}"  # k level
TIME_RANGE="${TIME_RANGE:-0 248}"  # plotted times, inclusive indices into the shared truth/emulator times
MAP_STEPPING="${MAP_STEPPING:-61}"  # map rows: every Nth time in TIME_RANGE
TS_STEPPING="${TS_STEPPING:-1}"  # time-series points: every Nth time in TIME_RANGE
MEAN_RANGE="${MEAN_RANGE:-0 248}"  # EKE time-mean window, inclusive indices
MEAN_STEPPING="${MEAN_STEPPING:-1}"
# ========================================

start=$(date +%s)
module load miniforge/24.3.0-0

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplconfig-${SLURM_JOB_ID:-$$}"
mkdir -p "$MPLCONFIGDIR"

echo "job=${SLURM_JOB_ID:-manual} node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}"
echo "emulators=${EMULATOR_LIST[*]} vars=$VARS mode=$MODE face=$FACE x=[$X_EXTENT) y=[$Y_EXTENT) k=$DEPTH t=[$TIME_RANGE] map_step=$MAP_STEPPING ts_step=$TS_STEPPING mean=[$MEAN_RANGE] step=$MEAN_STEPPING"

EMULATOR_ARGS=()
for e in "${EMULATOR_LIST[@]}"; do EMULATOR_ARGS+=(--emulator "$e"); done

# shellcheck disable=SC2086
/home/codycruz/LLC_ocean_emulator/high_res_diagnostics/.venv/bin/python -u /home/codycruz/LLC_ocean_emulator/coarse_diagnostics/coarse/scripts/KE_EKE_B.py \
  "${EMULATOR_ARGS[@]}" \
  --vars $VARS \
  --mode $MODE \
  --face "$FACE" \
  --x-extent $X_EXTENT \
  --y-extent $Y_EXTENT \
  --depth "$DEPTH" \
  --time-range $TIME_RANGE \
  --map-stepping "$MAP_STEPPING" \
  --ts-stepping "$TS_STEPPING" \
  --mean-range $MEAN_RANGE \
  --mean-stepping "$MEAN_STEPPING"

runtime=$(( $(date +%s) - start ))
printf "Total runtime: %02d:%02d:%02d\n" $((runtime/3600)) $((runtime%3600/60)) $((runtime%60))
