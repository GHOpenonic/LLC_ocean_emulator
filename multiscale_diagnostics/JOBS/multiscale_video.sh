#!/bin/bash
#SBATCH -p pi_abodner
#SBATCH --job-name=2026-11-24:multiscale:video-ep11-Agulhas_eighth_wide_shaprio_2m
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=00-06:00:00
#SBATCH --exclude=node2905
#SBATCH -o /home/codycruz/LLC_ocean_emulator/multiscale_diagnostics/logs/%x-%j.out
#SBATCH -e /home/codycruz/LLC_ocean_emulator/multiscale_diagnostics/logs/%x-%j.out
# Usage: sbatch multiscale_video.sh [full|fill]  (fill renders only missing frames; override any knob: REGION=globe VARS="KE" sbatch ...)

set -euo pipefail

# ============== PARAMETERS ==============
#EMULATOR="${EMULATOR:-ep11_shapiro=/orcd/scratch/seedfund/001/llc_emulator/multiscale/2026-11-15:multiscale_llc:face1-warm-lr1.5e-4/ema_ckpt_ep0011_2012-09-14T00_2012-11-14T00}"  # NAME=DIR (eval_multiscale's output_dir)
EMULATOR="${EMULATOR:-ep11_shapiro_agulhas=/orcd/scratch/seedfund/001/llc_emulator/multiscale//2026-11-15:multiscale_llc:face1-warm-lr1.5e-4/ema_ckpt_ep0011_2012-09-14T00_2012-11-14T00_agulhas}"
VARS="${VARS:-Theta B KE}"  # one row per var: any of Theta Salt U V Eta KE B
COLORSCALE="${COLORSCALE:-reg reg log}"  # log|reg per VARS entry, same order (log U/V is symlog)
DEPTH="${DEPTH:-0}"  # k level (Eta is always k=0)
TIME_RANGE="${TIME_RANGE:-0 1488}"  # inclusive hours since the IC
STEPPING="${STEPPING:-1}"  # hours; a store saved less often steps at its own interval and holds its last step between
# Map regions, degrees "lat0 lat1 lon0 lon1" (lon1 may pass 180); LAT_EXTENT / LON_EXTENT override.
declare -A REGIONS=(
  [globe]="-90 90 -180 180"
  [face1]="-57.01 13.74 -38 52"  # face 1 = N for the face-1 runs: all sharp
  [face1-wide]="-67 24 -48 62"  # face 1 with 10 deg of coarse around it
  [agulhas]="-48.39 -38.27 22 37"  # face 1 i[2880:3600) j[720:1440)
  [agulhas_wide]="-50 -36 20 39"  # face 1 i[2880:3600) j[720:1440)
  [agulhas_eighth_wide]="-39.96 -36.5 21 24.5"  # face 1 i[2880:3600) j[720:1440)
  [west_of_face1]="-57.01 -39 -49 -27"
  [west_zoom]="-39 -34 -41 -35"
)
REGION="${REGION:-agulhas_eighth_wide}"
[[ -n "${REGIONS[$REGION]:-}" ]] || { echo "unknown REGION=$REGION; one of: ${!REGIONS[*]}" >&2; exit 2; }
read -r _lat0 _lat1 _lon0 _lon1 <<< "${REGIONS[$REGION]}"
LAT_EXTENT="${LAT_EXTENT:-$_lat0 $_lat1}"
LON_EXTENT="${LON_EXTENT:-$_lon0 $_lon1}"
PIXELS="${PIXELS:-1000}"  # map width per panel in pixels
ERROR="${ERROR:-1}"  # 1 = add an emulator - truth column (title: RMSE over the shown coarse / sharp cells)
MIN_OCEAN_FRAC="${MIN_OCEAN_FRAC:-0.25}"  # mask cells under this ocean fraction at the level (coarse 1 - land_frac, sharp hFacC)
FPS="${FPS:-24}"
export VIDEO_N_WORKERS="${VIDEO_N_WORKERS:-40}"  # render processes, one frame each at a time
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

cat <<EOF
======== multiscale video ========
job           ${SLURM_JOB_ID:-manual}  node=$(hostname)  cpus=${SLURM_CPUS_PER_TASK:-?}  mem=${SLURM_MEM_PER_NODE:-?}  mode=$mode
emulator      $EMULATOR
vars          $VARS  (colorscale $COLORSCALE, k=$DEPTH)
time          hours [$TIME_RANGE] since the IC, every $STEPPING h (or a store's own interval if longer)
region        $REGION: lat [$LAT_EXTENT] lon [$LON_EXTENT], $PIXELS px wide
error col     $ERROR
ocean frac    >= $MIN_OCEAN_FRAC
video         $FPS fps, $VIDEO_N_WORKERS workers
==================================
EOF

ERROR_ARGS=()
[[ "$ERROR" == 1 ]] && ERROR_ARGS=(--error)

# shellcheck disable=SC2086
"$PY" -u /home/codycruz/LLC_ocean_emulator/multiscale_diagnostics/scripts/multiscale_video.py \
  --emulator "$EMULATOR" \
  --vars $VARS \
  --colorscale $COLORSCALE \
  --depth "$DEPTH" \
  --time-range $TIME_RANGE \
  --stepping "$STEPPING" \
  --lat-extent $LAT_EXTENT \
  --lon-extent $LON_EXTENT \
  --pixels "$PIXELS" \
  --min-ocean-frac "$MIN_OCEAN_FRAC" \
  --fps "$FPS" \
  --mode "$mode" \
  "${ERROR_ARGS[@]}"

runtime=$(( $(date +%s) - start ))
printf "Total runtime: %02d:%02d:%02d\n" $((runtime/3600)) $((runtime%3600/60)) $((runtime%60))
