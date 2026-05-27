#!/bin/bash 
###SBATCH -N 2
###SBATCH -t 01:00:00 
###SBATCH -t 3:00:00 
#SBATCH -J SGTDhfls 
#SBATCH --chdir=/home/smf/log/log_ml
#SBATCH --error=%x-%j.error 
#SBATCH --output=%x-%j.out
#SBATCH --gpus=1
#SBATCH --qos=ng
###SBATCH --qos=np
#SBATCH --mem-per-cpu=4G
###SBATCH --ntasks=8
###SBATCH --cpus-per-task=8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8

#DOMAIN='EmiliaRomagna'
DOMAIN='TestDomain'
VARIABLE='hfls'
#export CUDA_VISIBLE_DEVICES=1 

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
python3 main.py -c ../config/config_main_SG_${DOMAIN}_${VARIABLE}_atos.ini 

#cd $HOME/Script/Climulator
#python -m pytest

current_date_time="`date`";
echo The run ends at $current_date_time

exit 0 

