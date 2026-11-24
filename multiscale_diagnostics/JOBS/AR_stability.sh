#!/bin/bash
#SBATCH -p pi_abodner
#SBATCH --job-name=2026-11-23:multiscale:AR-stability-ep11-shapirovsnot
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --time=00-06:00:00
#SBATCH --exclude=node2905
#SBATCH -o /home/codycruz/LLC_ocean_emulator/multiscale_diagnostics/logs/%x-%j.out
#SBATCH -e /home/codycruz/LLC_ocean_emulator/multiscale_diagnostics/logs/%x-%j.out
# Usage: sbatch AR_stability.sh  (override any knob: EMULATORS="a=/dir/a b=/dir/b" MODE=SURFACE sbatch AR_stability.sh)

set -euo pipefail

# ============== PARAMETERS ==============
# One NAME=DIR per emulator; DIR holds sharp_pred.zarr and coarse_pred.zarr (eval_multiscale's output_dir).
DEFAULT_EMULATORS=(
  "ep11=/orcd/scratch/seedfund/001/llc_emulator/inference/multiscale/2026-11-15:multiscale_llc:face1-warm-lr1.5e-4/ema_ckpt_ep0011_2012-10-14T00_2012-11-14T00"
  "ep11-shapiro=/orcd/scratch/seedfund/001/llc_emulator/multiscale//2026-11-15:multiscale_llc:face1-warm-lr1.5e-4/ema_ckpt_ep0011_2012-10-14T00_2012-11-14T00"
)
if [[ -n "${EMULATORS:-}" ]]; then read -r -a EMULATOR_LIST <<< "$EMULATORS"; else EMULATOR_LIST=("${DEFAULT_EMULATORS[@]}"); fi
VARS="${VARS:-Theta Salt U V Eta}"
STATS="${STATS:-mean std rmse}"  # any of: mean std rmse (all listed are plotted)
MODE="${MODE:-FULL}"  # FULL (k x var grid) | COLUMN_MEAN_ONLY | SURFACE (k=0); the last two stack vars vertically
DEPTH_AVERAGE="${DEPTH_AVERAGE:-0 51 5}"  # column-mean levels: start stop step
DEPTH_LEVELS="${DEPTH_LEVELS:-10 25 50}"  # FULL mode only: rows after the column-mean and surface (k=0) rows, which are always drawn; don't list 0
TIME_RANGE="${TIME_RANGE:-0 744}"  # inclusive hours since the IC
STEPPING="${STEPPING:-6}"  # hours; a store saved less often steps at its own interval (coarse 3 h, sharp 1 or 3 h)
SHARP_BAND="${SHARP_BAND:-2160}"  # sharp rows per task (~4 GB per worker in FULL mode)
export AR_N_WORKERS="${AR_N_WORKERS:-32}"  # worker processes, one (step, band) task each
# ========================================

start=$(date +%s)
module load miniforge/24.3.0-0

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplconfig-${SLURM_JOB_ID:-$$}"
mkdir -p "$MPLCONFIGDIR"

cat <<EOF
======== multiscale AR stability ========
job           ${SLURM_JOB_ID:-manual}  node=$(hostname)  cpus=${SLURM_CPUS_PER_TASK:-?}  mem=${SLURM_MEM_PER_NODE:-?}
workers       $AR_N_WORKERS  (sharp band $SHARP_BAND rows)
vars          $VARS
stats         $STATS
mode          $MODE  (column k[$DEPTH_AVERAGE], levels [$DEPTH_LEVELS])
time          hours [$TIME_RANGE] since the IC, every $STEPPING h (or a store's own interval if longer)
EOF
for e in "${EMULATOR_LIST[@]}"; do printf 'emulator      %s\n' "$e"; done
echo "========================================="

EMULATOR_ARGS=()
for e in "${EMULATOR_LIST[@]}"; do EMULATOR_ARGS+=(--emulator "$e"); done

# shellcheck disable=SC2086
/home/codycruz/LLC_ocean_emulator/high_res_diagnostics/.venv/bin/python -u /home/codycruz/LLC_ocean_emulator/multiscale_diagnostics/scripts/AR_stability.py \
  "${EMULATOR_ARGS[@]}" \
  --vars $VARS \
  --stats $STATS \
  --mode "$MODE" \
  --depth-average $DEPTH_AVERAGE \
  --depth-levels $DEPTH_LEVELS \
  --time-range $TIME_RANGE \
  --stepping "$STEPPING" \
  --band "$SHARP_BAND"

runtime=$(( $(date +%s) - start ))
printf "Total runtime: %02d:%02d:%02d\n" $((runtime/3600)) $((runtime%3600/60)) $((runtime%60))
