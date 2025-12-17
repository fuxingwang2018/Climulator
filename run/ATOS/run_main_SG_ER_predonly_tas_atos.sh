#!/bin/bash 
###SBATCH -N 1
#SBATCH --nodes 1
#SBATCH -t 01:00:00 
#SBATCH -J SGERtas
#SBATCH --chdir=/home/smf/log/log_ml
#SBATCH --error=%x-%j.error 
#SBATCH --output=%x-%j.out
#SBATCH --gpus=1
#SBATCH --qos=ng
###SBATCH --mem-per-gpu=450G
#SBATCH --mem=80G
###SBATCH --ntasks=128
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1

###SBATCH --hint=nomultithread
###SBATCH -account=spselind

DOMAIN='EmiliaRomagna'
VARIABLE='tas'
export TF_GPU_ALLOCATOR=cuda_malloc_async
#export CUDA_VISIBLE_DEVICES=1 
export TF_DETERMINISTIC_OPS=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo 'domain is' ${DOMAIN}
#ecinteractive -g
module load netcdf4/4.9.2
#module load python3/new
module load cuda
source /perm/smf/venvs/climulator2/bin/activate

current_date_time="`date`";
echo The run starts from $current_date_time

set -exu 

cd $HOME/Scripts/Climulator/src
python3 main.py -c ../config/ATOS/config_main_SG_${DOMAIN}_${VARIABLE}_predonly_atos.ini 
#python3 test_time.py 

#cd $HOME/Script/Climulator
#python -m pytest

current_date_time="`date`";
echo The run ends at $current_date_time

exit 0 

