#!/bin/bash 
#SBATCH -A NAISS2025-1-11  -p alvis
#SBATCH -N 1 
###SBATCH --gpus-per-node=A40:1
#SBATCH --gpus-per-node=A100:4 
#SBATCH --cpus-per-task=64
#SBATCH -t 09:00:00
#SBATCH -J SGERtas
#SBATCH --chdir=/mimer/NOBACKUP/groups/naiss2025-6-138/HCLIMAI/log/log_ml/
#SBATCH --error=%x-%j.error 
#SBATCH --output=%x-%j.out

#export HDF5_USE_FILE_LOCKING=FALSE
#export TF_GPU_ALLOCATOR=cuda_malloc_async
##export CUDA_VISIBLE_DEVICES=1 
#export TF_DETERMINISTIC_OPS=0
#export TF_FORCE_GPU_ALLOW_GROWTH=true
#ecinteractive -g

DOMAIN='EmiliaRomagna'
#DOMAIN='TestDomain'
VARIABLE='tas'

echo 'domain is' ${DOMAIN}

module --force purge
#module load virtualenv/20.26.2-GCCcore-13.3.0
#module load Python/3.12.3-GCCcore-13.3.0
#module load netcdf4-python/1.7.1.post2-foss-2024a
module load virtualenv/20.23.1-GCCcore-12.3.0
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
module load netcdf4-python/1.6.4-foss-2023a
module load scikit-learn/1.4.2-gfbf-2023a
module load matplotlib/3.7.2-gfbf-2023a
source $HOME/venvs/climulator/bin/activate

current_date_time="`date`";
echo The run starts from $current_date_time
echo Check https://job.c3se.chalmers.se/alvis/$SLURM_JOB_ID for GPU usage.

set -exu 

cd $HOME/Climulator/src
python3 main.py -c ../config/ALVIS/config_main_SG_${DOMAIN}_${VARIABLE}_alvis.ini 

#cd $HOME/Climulator
#python -m pytest

current_date_time="`date`";
echo The run ends at $current_date_time

exit 0 

