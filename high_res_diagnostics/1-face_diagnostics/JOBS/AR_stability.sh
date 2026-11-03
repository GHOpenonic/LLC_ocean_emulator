#!/bin/bash
#SBATCH -p pi_abodner
#SBATCH --job-name=2026-11-01:1-f1-face:AR-stability-epoch17-rmse-column-mean-only-1
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=350G
#SBATCH --time=00-12:00:00
#SBATCH --exclude=node2905
#SBATCH -o /home/codycruz/LLC_ocean_emulator/high_res_diagnostics/1-face_diagnostics/logs/%x-%j.out
#SBATCH -e /home/codycruz/LLC_ocean_emulator/high_res_diagnostics/1-face_diagnostics/logs/%x-%j.out

set -euo pipefail

# ============== PARAMETERS ==============
EMULATOR_NAME="${EMULATOR_NAME:-1-face-ckpt17}"
EMULATOR_PATH="${EMULATOR_PATH:-/orcd/scratch/seedfund/001/llc_emulator/inference/1face//epoch17-ema/predictions.zarr}"
FACE="${FACE:-1}"  # LLC face of the truth; must match the emulator run
I_EXTENT="${I_EXTENT:-0 4320}"  # start stop (copied to i_g)
J_EXTENT="${J_EXTENT:-0 4320}"  # start stop (copied to j_g)
TIME_RANGE="${TIME_RANGE:-0 744}"  # inclusive indices
STEPPING="${STEPPING:-6}"
VARS="${VARS:-Theta Salt U V Eta}"
STATS="${STATS:-rmse}"  # any of: mean std rmse
MODE="${MODE:-SURFACE}"  # FULL (k x var grid) | COLUMN_MEAN_ONLY | SURFACE (k=0); the last two stack vars vertically
# ========================================

start=$(date +%s)
module load miniforge/24.3.0-0

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplconfig-${SLURM_JOB_ID:-$$}"
mkdir -p "$MPLCONFIGDIR"

echo "job=${SLURM_JOB_ID:-manual} node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}"
echo "emulator=$EMULATOR_NAME=$EMULATOR_PATH face=$FACE i=[$I_EXTENT) j=[$J_EXTENT) t=[$TIME_RANGE] step=$STEPPING vars=$VARS stats=$STATS mode=$MODE"

# shellcheck disable=SC2086
/home/codycruz/LLC_ocean_emulator/high_res_diagnostics/.venv/bin/python -u /home/codycruz/LLC_ocean_emulator/high_res_diagnostics/1-face_diagnostics/scripts/AR_stability.py \
  --emulator "$EMULATOR_NAME=$EMULATOR_PATH" \
  --face "$FACE" \
  --i-extent $I_EXTENT \
  --j-extent $J_EXTENT \
  --time-range $TIME_RANGE \
  --stepping "$STEPPING" \
  --vars $VARS \
  --stats $STATS \
  --mode "$(echo "$MODE" | tr 'A-Z_' 'a-z-')"

runtime=$(( $(date +%s) - start ))
printf "Total runtime: %02d:%02d:%02d\n" $((runtime/3600)) $((runtime%3600/60)) $((runtime%60))
