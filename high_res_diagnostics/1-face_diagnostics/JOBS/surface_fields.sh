#!/bin/bash
#SBATCH -p pi_abodner
#SBATCH --job-name=2026-11-24:warm_multiscale_face1:ckpt11_shapiro_2m_k=0
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=300G
#SBATCH --time=00-06:00:00
#SBATCH -o /home/codycruz/LLC_ocean_emulator/high_res_diagnostics/1-face_diagnostics/logs/%x-%j.out
#SBATCH -e /home/codycruz/LLC_ocean_emulator/high_res_diagnostics/1-face_diagnostics/logs/%x-%j.out

set -euo pipefail

# ============== PARAMETERS ==============
EMULATOR_NAME="${EMULATOR_NAME:-shapiro_warm_ckpt11_2m}"
EMULATOR_PATH="${EMULATOR_PATH:-/orcd/scratch/seedfund/001/llc_emulator/multiscale/2026-11-15:multiscale_llc:face1-warm-lr1.5e-4/ema_ckpt_ep0011_2012-09-14T00_2012-11-14T00/sharp_pred.zarr}"
FACE="${FACE:-1}"  # LLC face of the truth; must match the emulator run
I_EXTENT="${I_EXTENT:-0 4320}"  # start stop (copied to i_g)
J_EXTENT="${J_EXTENT:-0 4320}"  # start stop (copied to j_g)
TIME_RANGE="${TIME_RANGE:-0 1488}"  # inclusive indices
STEPPING="${STEPPING:-93}"
VARS="${VARS:-Theta Salt Eta U V}" # Salt U V Eta}"
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
echo "emulator=$EMULATOR_NAME=$EMULATOR_PATH face=$FACE i=[$I_EXTENT) j=[$J_EXTENT) t=[$TIME_RANGE] step=$STEPPING vars=$VARS"

# shellcheck disable=SC2086
/home/codycruz/LLC_ocean_emulator/high_res_diagnostics/.venv/bin/python -u /home/codycruz/LLC_ocean_emulator/high_res_diagnostics/1-face_diagnostics/scripts/surface_fields.py \
  --emulator "$EMULATOR_NAME=$EMULATOR_PATH" \
  --face "$FACE" \
  --i-extent $I_EXTENT \
  --j-extent $J_EXTENT \
  --time-range $TIME_RANGE \
  --stepping "$STEPPING" \
  --vars $VARS \
  --depth "$DEPTH"

runtime=$(( $(date +%s) - start ))
printf "Total runtime: %02d:%02d:%02d\n" $((runtime/3600)) $((runtime%3600/60)) $((runtime%60))
