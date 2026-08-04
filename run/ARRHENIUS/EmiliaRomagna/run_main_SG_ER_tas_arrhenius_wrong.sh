#!/bin/bash 
#SBATCH -A NAISS2026-4-912-gpu
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -J SGERtas
#SBATCH --chdir=/nobackup/proj/disk/hclimai/personal/fuxing/log/log_climulator/
#SBATCH --error=%x-%j.error 
#SBATCH --output=%x-%j.out
#SBATCH -p gpu
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=200G
######SBATCH --mem-per-cpu=10G 

#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=200G

###SBATCH -n 2
###SBATCH -c 72
###SBATCH --gpus 2
###SBATCH -t 10 # Requested walltime

#export HDF5_USE_FILE_LOCKING=FALSE
#export TF_GPU_ALLOCATOR=cuda_malloc_async
##export CUDA_VISIBLE_DEVICES=1 
#export TF_DETERMINISTIC_OPS=0
#export TF_FORCE_GPU_ALLOW_GROWTH=true
#ecinteractive -g

DOMAIN='EmiliaRomagna'
#DOMAIN='TestDomain'
VARIABLE='tas'
EXPNAME='wsmto_ERAI_2003'

echo 'domain is' ${DOMAIN}

module --force purge
#interactive -p gpu --gpus 1 -A NAISS2026-4-912-gpu
#module load GPU/buildenv-nvhpc/25.9-cu13.0
#module load GPU/Python/3.13.5-bare-gcc-2025b-eb
#cd $HOME/venvs
#python3 -m venv --system-site-packages climulator
#module load GPU/buildenv-gcccuda/2026.03-cu13.0

#source $HOME/venvs/climulator/bin/activate

#pip install tensorflow[and-cuda] (keras-3.14.1, tensorflow-2.21.0, nvidia-cublas-cu12-12.9.2.10)
#pip install scikit-learn (scikit-learn-1.9.0)
#pip install matplotlib (matplotlib-3.10.9)
#pip install netCDF4 (netCDF4-1.7.4)
#pip install pandas (pandas-3.0.3)
#pip install mlxtend (mlxtend-0.25.0)

#######
#$ module load GPU/Miniforge/26.3.2-2-eb
#$ module load GPU/Python/3.13.5-bare-gcc-2025b-eb
#$ mamba create -n climulator python=3.13 #--no-cache-dir
#$ mamba activate climulator
#$ mamba install -c conda-forge scikit-learn  #1.9.0
#$ mamba install -c conda-forge matplotlib  #3.10.9
#$ mamba install main::tensorflow-gpu #--no-cache-dir #2.21.0
#$ mamba install -c conda-forge netcdf4 #1.7.4
#$ mamba install -c conda-forge pandas #3.0.3
#$ mamba install -c conda-forge mlxtend #0.25.0
#$ mamba clean --all

module load GPU/Miniforge/26.3.2-2-eb
mamba activate climulator

current_date_time="`date`";
echo The run starts from $current_date_time
echo Check https://job.c3se.chalmers.se/alvis/$SLURM_JOB_ID for GPU usage.

set -exu 

cd $HOME/Climulator/src
#python3 main.py -c ../config/ARRHENIUS/config_main_SG_${DOMAIN}_${VARIABLE}_arrhenius.ini 
python3 main.py -c ../config/ARRHENIUS/config_main_SG_${DOMAIN}_${VARIABLE}_${EXPNAME}_arrhenius.ini 

#cd $HOME/Climulator
#python -m pytest

current_date_time="`date`";
echo The run ends at $current_date_time

exit 0 

