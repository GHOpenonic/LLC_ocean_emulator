#!/bin/bash
#SBATCH -p pi_abodner
#SBATCH --job-name=2026-11-03:1-face:surface-grads-hist-reg_vs_grad1_vs_bandh
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=36G
#SBATCH --time=00-06:00:00
#SBATCH --exclude=node2905
#SBATCH -o /home/codycruz/LLC_ocean_emulator/high_res_diagnostics/1-face_diagnostics/logs/%x-%j.out
#SBATCH -e /home/codycruz/LLC_ocean_emulator/high_res_diagnostics/1-face_diagnostics/logs/%x-%j.out

set -euo pipefail

# ============== PARAMETERS ==============
# One NAME=PATH per emulator; override with EMULATORS="a=/p/a.zarr b=/p/b.zarr".
DEFAULT_EMULATORS=(
  "reg17=/orcd/scratch/seedfund/001/llc_emulator/inference/1face/epoch17-ema/predictions.zarr"
  "reg12=/orcd/scratch/seedfund/001/llc_emulator/inference/1face/epoch12-ema-quintic/predictions.zarr"
  "GRAD1_12=/orcd/scratch/seedfund/001/llc_emulator/inference/1face/epoch12-ema-quintic-GRAD1/predictions.zarr"
  "bandH_11=/orcd/scratch/seedfund/001/llc_emulator/inference/1face/epoch11-band-H/predictions.zarr"
)
if [[ -n "${EMULATORS:-}" ]]; then read -r -a EMULATOR_LIST <<< "$EMULATORS"; else EMULATOR_LIST=("${DEFAULT_EMULATORS[@]}"); fi
FACE="${FACE:-1}"  # LLC face of the truth; must match the emulator run
# I_EXTENT="${I_EXTENT:-0 4320}"  # start stop (copied to i_g)
# J_EXTENT="${J_EXTENT:-0 4320}"  # start stop (copied to j_g)
I_EXTENT="${I_EXTENT:-2880 3600}"  # start stop (copied to i_g)
J_EXTENT="${J_EXTENT:-720 1440}"  # start stop (copied to j_g)
TIME_RANGE="${TIME_RANGE:-0 335}"  # inclusive indices
STEPPING="${STEPPING:-42}"
VARS="${VARS:-Theta Salt U V Eta}"
DEPTH="${DEPTH:-0}"  # k level
# ========================================

start=$(date +%s)
module load miniforge/24.3.0-0

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplconfig-${SLURM_JOB_ID:-$$}"
mkdir -p "$MPLCONFIGDIR"

echo "job=${SLURM_JOB_ID:-manual} node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}"
echo "emulators=${EMULATOR_LIST[*]} face=$FACE i=[$I_EXTENT) j=[$J_EXTENT) t=[$TIME_RANGE] step=$STEPPING vars=$VARS"

EMULATOR_ARGS=()
for e in "${EMULATOR_LIST[@]}"; do EMULATOR_ARGS+=(--emulator "$e"); done

# shellcheck disable=SC2086
/home/codycruz/LLC_ocean_emulator/high_res_diagnostics/.venv/bin/python -u /home/codycruz/LLC_ocean_emulator/high_res_diagnostics/1-face_diagnostics/scripts/surface_gradients_hist.py \
  "${EMULATOR_ARGS[@]}" \
  --face "$FACE" \
  --i-extent $I_EXTENT \
  --j-extent $J_EXTENT \
  --time-range $TIME_RANGE \
  --stepping "$STEPPING" \
  --vars $VARS \
  --depth "$DEPTH"

runtime=$(( $(date +%s) - start ))
printf "Total runtime: %02d:%02d:%02d\n" $((runtime/3600)) $((runtime%3600/60)) $((runtime%60))
