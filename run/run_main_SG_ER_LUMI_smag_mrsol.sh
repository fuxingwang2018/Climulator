#!/bin/bash 
###SBATCH --partition=standard-g  # Change this based on LUMI-G configuration
#SBATCH --partition=small-g  # Change this based on LUMI-G configuration
###SBATCH --partition=dev-g  # Change this based on LUMI-G configuration
###SBATCH --nodes=2
#SBATCH --time=08:00:00
#SBATCH -A project_465000527
#SBATCH --chdir=/users/wangfuxi/log
#SBATCH --error=%x-%j.error
#SBATCH --output=%x-%j.out
#SBATCH -J SGERmrsol 
#SBATCH --gpus=4
###SBATCH --ntasks-per-node=8     # 8 MPI ranks per node, 16 total (2x8)
###SBATCH --gpus-per-node=8      # Allocate one gpu per MPI rank
#SBATCH --mem-per-gpu=100G

DOMAIN='EmiliaRomagna'
#DOMAIN='TestDomain'
VARIABLE='mrsol'
echo 'domain and variable:' ${DOMAIN}, ${VARIABLE}

#
#module load LUMI/24.03  # Load the correct LUMI module
source $HOME/venvs/Climulator/bin/activate
#module load cray-python/3.11.7
module load cray-python/3.10.10

module load Local-CSC
#available version tensorflow/2.8, 2.9, 2.10, 2.11, 2.12. 2.16, but tensorflow needs to be consistent with keras which is keras==2.12.0
module load tensorflow/2.12

current_date_time="`date`";
echo The run starts from $current_date_time

set -exu 

cd $HOME/Scripts/Climulator/src
python3 main.py -c ../config/config_main_SG_${DOMAIN}_${VARIABLE}_LUMI.ini 

current_date_time="`date`";
echo The run ends at $current_date_time

exit 0 


