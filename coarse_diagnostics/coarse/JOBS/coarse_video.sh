#!/bin/bash
#SBATCH -p pi_abodner
#SBATCH --job-name=2026-11-07:coarse:3hr-gulfstream-ALLTIME-KE-B-log
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=00-04:00:00
#SBATCH -o /home/codycruz/LLC_ocean_emulator/coarse_diagnostics/coarse/logs/%x-%j.out
#SBATCH -e /home/codycruz/LLC_ocean_emulator/coarse_diagnostics/coarse/logs/%x-%j.out

set -euo pipefail

# Usage:
#   sbatch coarse_video.sh         # render every frame, then encode the MP4
#   sbatch coarse_video.sh fill    # render only missing frames, then encode

# ============== PARAMETERS ==============
# One NAME=PATH per emulator, columns in this order; override with EMULATORS="a=/p/a.zarr b=/p/b.zarr".
DEFAULT_EMULATORS=(
  "w384_3hr_eager_ckpt36-ALLTIME=/orcd/scratch/seedfund/001/llc_emulator/inference/coarse//w384-ckpt36-3hr-eager-ALLTIME/predictions.zarr"
)

if [[ -n "${EMULATORS:-}" ]]; then read -r -a EMULATOR_LIST <<< "$EMULATORS"; else EMULATOR_LIST=("${DEFAULT_EMULATORS[@]}"); fi
VIDEO_VARS="${VIDEO_VARS:-KE B}"  # one row per var: any of Theta Salt U V Eta KE B
COLORSCALE="${COLORSCALE:-log log}"  # log|reg per VIDEO_VARS entry, same order (log U/V is symlog)
VIDEO_K="${VIDEO_K:-0}"  # depth level
FPS="${FPS:-48}"
TIME_RANGE="${TIME_RANGE:-0 1712}"  # inclusive indices into the shared truth/emulator times
STEPPING="${STEPPING:-1}"
FACE="${FACE:-10}"  # 0-12 for one face, all for the lat/lon globe
X_EXTENT="${X_EXTENT:-135 225}"  # start stop, per face
Y_EXTENT="${Y_EXTENT:-45 95}"  # start stop, per face
# X_EXTENT="${X_EXTENT:-180 225}"  # start stop, per face
# Y_EXTENT="${Y_EXTENT:-45 90}"  # start stop, per face
export VIDEO_N_WORKERS="${VIDEO_N_WORKERS:-48}"  # render processes
# ========================================

mode="${1:-full}"
case "$mode" in
  full|fill) ;;
  *) echo "Usage: sbatch $0 [full|fill]" >&2; exit 2 ;;
esac

start=$(date +%s)
module load miniforge/24.3.0-0  # provides ffmpeg

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplconfig-${SLURM_JOB_ID:-$$}"
mkdir -p "$MPLCONFIGDIR"

PY=/home/codycruz/LLC_ocean_emulator/high_res_diagnostics/.venv/bin/python
"$PY" -c "import matplotlib.pyplot as plt; plt.figure()" >/dev/null 2>&1 || true  # warm font cache once

echo "job=${SLURM_JOB_ID:-manual} node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?} mode=$mode"
echo "emulators=${EMULATOR_LIST[*]} vars=$VIDEO_VARS scale=$COLORSCALE k=$VIDEO_K fps=$FPS t=[$TIME_RANGE] step=$STEPPING face=$FACE x=[$X_EXTENT) y=[$Y_EXTENT) workers=$VIDEO_N_WORKERS"

EMULATOR_ARGS=()
for e in "${EMULATOR_LIST[@]}"; do EMULATOR_ARGS+=(--emulator "$e"); done

# shellcheck disable=SC2086
"$PY" -u /home/codycruz/LLC_ocean_emulator/coarse_diagnostics/coarse/scripts/coarse_video.py \
  "${EMULATOR_ARGS[@]}" \
  --vars $VIDEO_VARS \
  --colorscale $COLORSCALE \
  --depth "$VIDEO_K" \
  --fps "$FPS" \
  --time-range $TIME_RANGE \
  --stepping "$STEPPING" \
  --face "$FACE" \
  --x-extent $X_EXTENT \
  --y-extent $Y_EXTENT \
  --mode "$mode"

runtime=$(( $(date +%s) - start ))
printf "Total runtime: %02d:%02d:%02d\n" $((runtime/3600)) $((runtime%3600/60)) $((runtime%60))
