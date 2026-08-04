#!/bin/bash 
#SBATCH -A NAISS2026-4-912-gpu
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -J SGTDtas
#SBATCH --chdir=/nobackup/proj/disk/hclimai/personal/fuxing/log/log_climulator/
#SBATCH --error=%x-%j.error 
#SBATCH --output=%x-%j.out
#SBATCH -p gpu
##SBATCH -n 1 
#SBATCH --cpus-per-task=16
##SBATCH --gpus 1
#SBATCH --mem-per-gpu=400G

###SBATCH -c 48
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
###SBATCH --mem-per-cpu=10G 

#export TF_DETERMINISTIC_OPS=1
#export CUDA_VISIBLE_DEVICES=1 
export CUDA_VISIBLE_DEVICES=0  # Optional: to fix device ordering
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_DETERMINISTIC_OPS=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

#set -exu 

current_date_time="`date`";
echo The run starts from $current_date_time

DOMAIN='TestDomain'
VARIABLE='tas'
GCM='ERAI'
echo 'domain and variable:' ${DOMAIN}, ${VARIABLE}

CONTAINER=/nobackup/proj/disk/hclimai/personal/fuxing/containers/tensorflow_25.02-tf2-py3.sif
SRC=$HOME/Climulator/src
#export PYTHONPATH=/home/fuxing/container_packages:$PYTHONPATH
export PYTHONPATH=/home/fuxing/Climulator:/home/fuxing/container_packages:${PYTHONPATH:-}
cd $HOME/Climulator

apptainer run --nv \
    --bind /home/fuxing:/home/fuxing \
    --bind /nobackup/proj/disk/hclimai:/nobackup/proj/disk/hclimai \
    $CONTAINER \
    python3 $SRC/main.py -c /home/fuxing/Climulator/config/ARRHENIUS/${DOMAIN}/config_main_SG_${DOMAIN}_${VARIABLE}_arrhenius.ini 


#cd /nobackup/proj/disk/hclimai/personal/fuxing/
#mkdir -p containers
#cd containers
#apptainer pull docker://nvcr.io/nvidia/tensorflow:25.02-tf2-py3
#apptainer run --nv tensorflow_25.02-tf2-py3.sif     python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
#apptainer run --nv tensorflow_25.02-tf2-py3.sif     pip install matplotlib --no-deps     --target /home/fuxing/container_packages
#apptainer run --nv tensorflow_25.02-tf2-py3.sif     pip install pillow --no-deps --target /home/fuxing/container_packages
#apptainer run --nv tensorflow_25.02-tf2-py3.sif     pip install cycler --no-deps --target /home/fuxing/container_packages
#apptainer run --nv tensorflow_25.02-tf2-py3.sif     pip install kiwisolver fonttools contourpy pyparsing python-dateutil --no-deps --target /home/fuxing/container_packages
#apptainer run --nv tensorflow_25.02-tf2-py3.sif     pip install netcdf --no-deps --target /home/fuxing/container_packages
#apptainer run --nv tensorflow_25.02-tf2-py3.sif     pip install cftime --no-deps --target /home/fuxing/container_packages
#apptainer run --nv tensorflow_25.02-tf2-py3.sif     pip install "mlxtend<0.23.0" --no-deps --target /home/fuxing/container_packages --force-reinstall

#module --force purge
#module load GPU/Miniforge/26.3.2-2-eb
#mamba activate climulator

###module load GPU/buildtool-easybuild/5.2.1-hpca3ef7d197
###module load CUDA/12.9.1
###source $HOME/venvs/climulator/bin/activate

#python3 main.py -c ../config/ARRHENIUS/config_main_SG_${DOMAIN}_${VARIABLE}_arrhenius.ini 

current_date_time="`date`";
echo The run ends at $current_date_time

exit 0 

