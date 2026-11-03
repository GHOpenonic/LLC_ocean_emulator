#!/bin/bash
#SBATCH -p pi_abodner
#SBATCH --job-name=2026-11-01:1-face-3D-full-ckpt17
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --mem=360G
#SBATCH --time=00-10:00:00
#SBATCH -o /home/codycruz/LLC_ocean_emulator/high_res_diagnostics/videos/logs/%x-%j.out
#SBATCH -e /home/codycruz/LLC_ocean_emulator/high_res_diagnostics/videos/logs/%x-%j.out
#SBATCH --nodelist node4006,node4005


set -euo pipefail

# ============== PARAMETERS ==============
export VIDEO_EMULATOR_NAME="${VIDEO_EMULATOR_NAME:-1-face-ckpt17}"
export VIDEO_EMULATOR_PATH="${VIDEO_EMULATOR_PATH:-\
/orcd/scratch/seedfund/001/llc_emulator/inference/1face//epoch17-ema/predictions.zarr}"
export VIDEO_I_EXTENT="${VIDEO_I_EXTENT:-0 4320}"  # start stop (copied to i_g) # agulhas 2880 3600
export VIDEO_J_EXTENT="${VIDEO_J_EXTENT:-0 4320}"  # start stop (copied to j_g) # agulhas 720 1440
export VIDEO_TIME_RANGE="${VIDEO_TIME_RANGE:-0 744}"  # inclusive indices
export VIDEO_STEPPING="${VIDEO_STEPPING:-4}"
export VIDEO_VARS="${VIDEO_VARS:-Theta}"  # rows, e.g. "Theta Salt U V"
export VIDEO_GRADIENTS="${VIDEO_GRADIENTS:-0}"  # 1 = add gradient panel after each model (|∇h| top, |∂z| walls)
export VIDEO_ERRORS="${VIDEO_ERRORS:-0}"  # 1 = add emulator - llc panel at the end of each row
export VIDEO_REGION_NAME="${VIDEO_REGION_NAME:-1-face i[$VIDEO_I_EXTENT) j[$VIDEO_J_EXTENT)}"  # frame title
# ========================================

start=$(date +%s)

# load module
module load miniforge/24.3.0-0

# set location of script
location=/home/codycruz/LLC_ocean_emulator/high_res_diagnostics/videos

# Job type flag
job_type="llc_3D-4-var-video_parallel"

# Usage:
#   sbatch video_gen.sh              # render every frame, then encode the MP4
#   sbatch video_gen.sh fill         # render only missing frames, then encode
#   sbatch video_gen.sh full --gif   # also write a GIF (2 extra decode passes,
#                                    # and far larger than the MP4; off by default)
#
# For a new emulator/variable/time config, prime the face cache in its own job
# first. Extracting the faces reads a few hundred GiB (the walls need every k
# level, and an emulator chunk is a whole 720x720 plane), so it wants a long
# wall clock but hardly any cores -- and it caches per variable, so a job that
# dies part way through keeps what it already finished:
#   sbatch --cpus-per-task=16 --time=12:00:00 video_gen.sh full --preload-only
# After that the render starts from cache in seconds:
#   sbatch video_gen.sh
mode="${1:-full}"
case "$mode" in
  full|fill) shift || true ;;
  *)
    echo "Usage: sbatch $0 [full|fill] [extra args for the python script]" >&2
    exit 2
    ;;
esac

echo "Job: $job_type (mode: $mode)"

# ── Runtime environment ───────────────────────────────────────────────
# One BLAS thread per worker. With ~100 workers, per-worker thread pools
# oversubscribe the node ~100x and the context switching eats the run.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# glibc hands every thread a 64 MiB malloc arena; across ~100 forked workers
# that is pure RSS inflation.
export MALLOC_ARENA_MAX=2

# Render workers. 100 x the per-worker footprint overran --mem=300G (OOM kills
# left the pool waiting forever); 40 fits with headroom.
export VIDEO_N_WORKERS="${VIDEO_N_WORKERS:-40}"

# Matplotlib's font cache lives in $HOME by default. 100 processes stat-ing
# (and racing to rebuild) it on a shared filesystem is a slow, flaky start;
# give this job its own cache on node-local scratch instead.
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplconfig-${SLURM_JOB_ID:-$$}"
mkdir -p "$MPLCONFIGDIR"

# activate virtual environment
source /home/codycruz/Ocean_Emulator/.venv/bin/activate

echo "======== create video (3D, ${VIDEO_N_WORKERS} render workers) ========"

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "NODE=$(hostname)"
echo "CPUS=$SLURM_CPUS_PER_TASK"
echo "MEM=$SLURM_MEM_PER_NODE"

# Warm the font cache once, serially, so the workers never contend over it.
python -c "import matplotlib.pyplot as plt; plt.figure()" >/dev/null 2>&1 || true

uv run "$location/${job_type}.py" --mode "$mode" "$@"

end=$(date +%s)
runtime=$((end-start))
printf "Total runtime: %02d:%02d:%02d\n" \
  $((runtime/3600)) $((runtime%3600/60)) $((runtime%60))

echo "======== job complete ========"
