#!/bin/bash
#SBATCH -p pi_abodner
#SBATCH --job-name=2026-11-12:coarse:AR-stability-1month-384vs640
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=600G
#SBATCH --time=00-04:00:00
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
VARS="${VARS:-Theta Salt U V Eta}"
STATS="${STATS:-mean std rmse}"  # any of: mean std rmse (all listed are plotted)
MODE="${MODE:-FULL}"  # FULL (k x var grid) | COLUMN_MEAN_ONLY | SURFACE (k=0); the last two stack vars vertically
DEPTH_AVERAGE="${DEPTH_AVERAGE:-0 51 5}"  # column-mean levels: start stop step
DEPTH_LEVELS="${DEPTH_LEVELS:-10 25 50}"  # extra single-level rows in FULL mode (k=0 is the 'surface' row)
TIME_BLOCK="${TIME_BLOCK:-4}"  # time steps per worker task (~1 GB per source per step on the globe)
TIME_RANGE="${TIME_RANGE:-0 248}"  # inclusive indices into the shared truth/emulator times
STEPPING="${STEPPING:-1}"
FACE="${FACE:-all}"  # 0-12 for one face, all for the globe
X_EXTENT="${X_EXTENT:-0 270}"  # start stop, per face 0 270
Y_EXTENT="${Y_EXTENT:-0 270}"  # start stop, per face
export AR_N_WORKERS="${AR_N_WORKERS:-16}"  # worker processes; each holds TIME_BLOCK steps of every source
# ========================================

start=$(date +%s)
module load miniforge/24.3.0-0

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplconfig-${SLURM_JOB_ID:-$$}"
mkdir -p "$MPLCONFIGDIR"

echo "job=${SLURM_JOB_ID:-manual} node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?} workers=$AR_N_WORKERS"
echo "emulators=${EMULATOR_LIST[*]} vars=$VARS stats=$STATS mode=$MODE davg=[$DEPTH_AVERAGE] levels=[$DEPTH_LEVELS] block=$TIME_BLOCK t=[$TIME_RANGE] step=$STEPPING face=$FACE x=[$X_EXTENT) y=[$Y_EXTENT)"

EMULATOR_ARGS=()
for e in "${EMULATOR_LIST[@]}"; do EMULATOR_ARGS+=(--emulator "$e"); done

# shellcheck disable=SC2086
/home/codycruz/LLC_ocean_emulator/high_res_diagnostics/.venv/bin/python -u /home/codycruz/LLC_ocean_emulator/coarse_diagnostics/coarse/scripts/AR_stability.py \
  "${EMULATOR_ARGS[@]}" \
  --vars $VARS \
  --stats $STATS \
  --mode "$MODE" \
  --depth-average $DEPTH_AVERAGE \
  --depth-levels $DEPTH_LEVELS \
  --time-block "$TIME_BLOCK" \
  --time-range $TIME_RANGE \
  --stepping "$STEPPING" \
  --face "$FACE" \
  --x-extent $X_EXTENT \
  --y-extent $Y_EXTENT

runtime=$(( $(date +%s) - start ))
printf "Total runtime: %02d:%02d:%02d\n" $((runtime/3600)) $((runtime%3600/60)) $((runtime%60))
