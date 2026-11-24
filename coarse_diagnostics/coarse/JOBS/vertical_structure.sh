#!/bin/bash
#SBATCH -p pi_abodner
#SBATCH --job-name=2026-11-05:coarse:vertical-structure-SO
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=00-02:00:00
#SBATCH -o /home/codycruz/LLC_ocean_emulator/coarse_diagnostics/coarse/logs/%x-%j.out
#SBATCH -e /home/codycruz/LLC_ocean_emulator/coarse_diagnostics/coarse/logs/%x-%j.out

set -euo pipefail

# ============== PARAMETERS ==============
# One NAME=PATH per emulator, columns in this order; override with EMULATORS="a=/p/a.zarr b=/p/b.zarr".
DEFAULT_EMULATORS=(
  "w384-ckpt48=/orcd/scratch/seedfund/001/llc_emulator/inference/coarse/w384-ckpt48/predictions.zarr"
  "w512-ckpt47=/orcd/scratch/seedfund/001/llc_emulator/inference/coarse//w512-ckpt47/predictions.zarr"
  "coarsen(1-face_ckpt17)=/orcd/scratch/seedfund/001/llc_emulator/inference/1face/epoch17-ema/predictions_third_coarse.zarr"
)

if [[ -n "${EMULATORS:-}" ]]; then read -r -a EMULATOR_LIST <<< "$EMULATORS"; else EMULATOR_LIST=("${DEFAULT_EMULATORS[@]}"); fi
VARS="${VARS:-Theta}"  # 3D vars only
K="${K:-0 51}"  # start stop
# DOMAIN: the one given as "START STOP" is the transect axis, the single value is the constant
# LAT="${LAT:--90 90}"  # or -60 for a lon transect at constant lat
# LON="${LON:--25}"  # or "0 360"
LAT="${LAT:--60}" 
LON="${LON:--180 180}"  
RES="${RES:-0.3333333}"  # transect sample spacing [deg]
TIME_RANGE="${TIME_RANGE:-0 731}"  # inclusive indices into the shared truth/emulator times
STEPPING="${STEPPING:-93}"
# ========================================

start=$(date +%s)
module load miniforge/24.3.0-0

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplconfig-${SLURM_JOB_ID:-$$}"
mkdir -p "$MPLCONFIGDIR"

echo "job=${SLURM_JOB_ID:-manual} node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}"
echo "emulators=${EMULATOR_LIST[*]} vars=$VARS k=[$K) lat=[$LAT] lon=[$LON] res=$RES t=[$TIME_RANGE] step=$STEPPING"

EMULATOR_ARGS=()
for e in "${EMULATOR_LIST[@]}"; do EMULATOR_ARGS+=(--emulator "$e"); done

# shellcheck disable=SC2086
/home/codycruz/LLC_ocean_emulator/high_res_diagnostics/.venv/bin/python -u /home/codycruz/LLC_ocean_emulator/coarse_diagnostics/coarse/scripts/vertical_structure.py \
  "${EMULATOR_ARGS[@]}" \
  --vars $VARS \
  --k $K \
  --lat $LAT \
  --lon $LON \
  --res "$RES" \
  --time-range $TIME_RANGE \
  --stepping "$STEPPING"

runtime=$(( $(date +%s) - start ))
printf "Total runtime: %02d:%02d:%02d\n" $((runtime/3600)) $((runtime%3600/60)) $((runtime%60))
