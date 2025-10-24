#!/bin/bash 
###SBATCH -N 1 
###SBATCH -t 01:00:00 
###SBATCH -t 3:00:00 
#SBATCH -J SG _mrosl
#SBATCH --chdir=/home/smf/log/log_ml
#SBATCH --error=%x-%j.error 
#SBATCH --output=%x-%j.out
#SBATCH --gpus=1
#SBATCH --qos=ng
###SBATCH --qos=nf
#SBATCH --mem-per-cpu=2G
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8

DOMAIN='EmiliaRomagna'
#DOMAIN='TestDomain'
#export TF_GPU_ALLOCATOR=cuda_malloc_async
#export CUDA_VISIBLE_DEVICES=1 
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo 'domain is' ${DOMAIN}
#ecinteractive -g
module load netcdf4/4.9.2
#module load python3/new
module load cuda
source /perm/smf/venvs/climulator2/bin/activate

current_date_time="`date`";
echo The run starts from $current_date_time

#source ~sm_fuxwa/anaconda3/bin/activate
#source /nobackup/rossby24/users/sm_fuxwa/conda/miniconda3/bin/activate
#conda activate hclimai

set -exu 

cd $HOME/Scripts/HCLIMAI/src
python3 main.py -c ../config/config_main_SG_${DOMAIN}_mrsol.ini 

#cd $HOME/Script/HCLIMAI
#python -m pytest

current_date_time="`date`";
echo The run ends at $current_date_time

exit 0 

