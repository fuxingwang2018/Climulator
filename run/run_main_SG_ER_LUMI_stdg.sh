#!/bin/bash 
#SBATCH --partition=standard-g  # Change this based on LUMI-G configuration
###SBATCH --partition=small-g  # Change this based on LUMI-G configuration
###SBATCH --partition=dev-g  # Change this based on LUMI-G configuration
#SBATCH --nodes=1
#SBATCH --time=00:30:00  # 10 minutes
#SBATCH -A project_465000527
#SBATCH --chdir=/users/wangfuxi/log
#SBATCH --error=%x-%j.error
#SBATCH --output=%x-%j.out
#SBATCH -J SGERstd 
###SBATCH --gpus=1
###SBATCH --ntasks-per-node=4     # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --gpus-per-node=8      # Allocate one gpu per MPI rank
###SBATCH --cpus-per-task=10      # Number of CPU cores per task
#SBATCH --mem-per-gpu=60G
###SBATCH --mem=400G
#
DOMAIN='EmiliaRomagna'
#DOMAIN='TestDomain'
echo 'domain is' ${DOMAIN}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export LD_PRELOAD=""
#export TF_CUDNN_WORKSPACE_LIMIT_IN_MB=81920

export TF_FORCE_GPU_ALLOW_GROWTH=true

export OMP_NUM_THREADS=8
#export MPICH_GPU_SUPPORT_ENABLED=1

ulimit -c 0  # Disable core dumps for the session

#
#module load LUMI/24.03  # Load the correct LUMI module
source $HOME/venvs/Climulator/bin/activate
#module load cray-python/3.11.7
module load cray-python/3.10.10

module load Local-CSC
#available version tensorflow/2.8, 2.9, 2.10, 2.11, 2.12. 2.16, but tensorflow needs to be consistent with keras which is keras==2.12.0
module load tensorflow/2.12
module load craype-accel-amd-gfx90a
module load cray-mpich/8.1.28

current_date_time="`date`";
echo The run starts from $current_date_time

set -exu 

cd $HOME/Scripts/Climulator/src
python3 main.py -c ../config/config_main_SG_${DOMAIN}_LUMI.ini 
#srun --gpu-bind=map_gpu:0,1,2,3 python3 main.py -c ../config/config_main_SG_${DOMAIN}_LUMI.ini 

current_date_time="`date`";
echo The run ends at $current_date_time

exit 0 


