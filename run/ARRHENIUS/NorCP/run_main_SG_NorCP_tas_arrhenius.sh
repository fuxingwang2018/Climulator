#!/bin/bash 
#SBATCH -A NAISS2026-4-912-gpu
###SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 04:00:00
#SBATCH -J SGNCPtas
#SBATCH --chdir=/nobackup/proj/disk/hclimai/personal/fuxing/log/log_climulator/
#SBATCH --error=%x-%j.error 
#SBATCH --output=%x-%j.out
#SBATCH -p gpu
###SBATCH --ntasks-per-node=1 #4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
###SBATCH --mem-per-gpu=400G
###SBATCH --mem-per-cpu=10G 
###SBATCH --exclusive
#SBATCH --mem=400G

###SBATCH --ntasks-per-node=4
###SBATCH --cpus-per-task=32
###SBATCH --gpus-per-node=1
###SBATCH --mem-per-gpu=200G

###SBATCH -n 2
###SBATCH -c 72
###SBATCH --gpus 2
###SBATCH -t 10 # Requested walltime

#export HDF5_USE_FILE_LOCKING=FALSE
export TF_GPU_ALLOCATOR=cuda_malloc_async
export CUDA_VISIBLE_DEVICES=0 #1 
export TF_DETERMINISTIC_OPS=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
#ecinteractive -g

#set -exu 
#module --force purge
#interactive -p gpu --gpus 1 -A NAISS2026-4-912-gpu
#module load GPU/buildenv-nvhpc/25.9-cu13.0
#module load GPU/Python/3.13.5-bare-gcc-2025b-eb
#python3 -m venv --system-site-packages climulator
#module load GPU/buildenv-gcccuda/2026.03-cu13.0

current_date_time="`date`";
echo The run starts from $current_date_time
echo Check https://job.c3se.chalmers.se/alvis/$SLURM_JOB_ID for GPU usage.

DOMAIN='NorCP'
#DOMAIN='EmiliaRomagna'
#DOMAIN='TestDomain'
VARIABLE='tas'
echo 'domain is' ${DOMAIN}

CONTAINER=/nobackup/proj/disk/hclimai/personal/fuxing/containers/tensorflow_25.02-tf2-py3.sif
SRC=$HOME/Climulator/src
#export PYTHONPATH=/home/fuxing/container_packages:$PYTHONPATH
export PYTHONPATH=/home/fuxing/Climulator:/home/fuxing/container_packages:${PYTHONPATH:-}
cd $HOME/Climulator

apptainer run --nv \
    --bind /home/fuxing:/home/fuxing \
    --bind /nobackup/proj/disk/hclimai:/nobackup/proj/disk/hclimai \
    $CONTAINER \
    python3 $SRC/main.py -c /home/fuxing/Climulator/config/ARRHENIUS/NorCP/config_main_SG_${DOMAIN}_${VARIABLE}_arrhenius.ini 

#apptainer run --nv tensorflow_25.02-tf2-py3.sif     pip install matplotlib --no-deps     --target /home/fuxing/container_packages
#apptainer run --nv tensorflow_25.02-tf2-py3.sif     pip install pillow --no-deps --target /home/fuxing/container_packages
#apptainer run --nv tensorflow_25.02-tf2-py3.sif     pip install cycler --no-deps --target /home/fuxing/container_packages
#apptainer run --nv tensorflow_25.02-tf2-py3.sif     pip install kiwisolver fonttools contourpy pyparsing python-dateutil --no-deps --target /home/fuxing/container_packages
#apptainer run --nv tensorflow_25.02-tf2-py3.sif     pip install netcdf --no-deps --target /home/fuxing/container_packages
#apptainer run --nv tensorflow_25.02-tf2-py3.sif     pip install cftime --no-deps --target /home/fuxing/container_packages
#apptainer run --nv tensorflow_25.02-tf2-py3.sif     pip install "mlxtend<0.23.0" --no-deps --target /home/fuxing/container_packages --force-reinstall

current_date_time="`date`";
echo The run ends at $current_date_time

exit 0 

