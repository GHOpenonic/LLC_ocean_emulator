#!/bin/bash
#SBATCH -p pi_abodner
#SBATCH --job-name=llc_emulator_3D-theta-video-full-ckpt40-parallel
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=850GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --time=00-24:00:00
#SBATCH -o /home/codycruz/LLC_ocean_emulator/high_res_diagnostics/videos/logs/%x-%j.out
#SBATCH -e /home/codycruz/LLC_ocean_emulator/high_res_diagnostics/videos/logs/%x-%j.out

start=$(date +%s)

# load module
module load miniforge/24.3.0-0

# set location of script
location=/home/codycruz/LLC_ocean_emulator/high_res_diagnostics/videos

# Job type flag
job_type="llc_3D-4-var-video_parallel" 

echo "Job:$job_type"

# activate virtual environment
source /home/codycruz/Ocean_Emulator/.venv/bin/activate


echo "======== create gif (4 vars, 3D) ========"

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "CPUS=$SLURM_CPUS_PER_TASK"
echo "MEM=$SLURM_MEM_PER_NODE"

uv run "$location/${job_type}.py"

end=$(date +%s)
runtime=$((end-start))
printf "Total runtime: %02d:%02d:%02d\n" \
  $((runtime/3600)) $((runtime%3600/60)) $((runtime%60))

echo "======== job complete ========"
