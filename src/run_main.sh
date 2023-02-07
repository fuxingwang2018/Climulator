#!/bin/bash 
#SBATCH -N 1 
#SBATCH -t 6:00:00 
####SBATCH -n 1  ##ntasks 
###SBATCH --mem=256000 
#SBATCH -J SRGAN 
#SBATCH --chdir=/nobackup/rossby26/users/sm_fuxwa/AI/log
#SBATCH --error=%x-%j.error 
#SBATCH --output=%x-%j.out

module load netCDF/4.3.2-HDF5-1.8.12-nsc1-intel-2018.u1-bare
module load HDF5/1.8.12-nsc1-intel-2018.u1-bare

current_date_time="`date`";
echo The run starts from $current_date_time

source ~sm_fuxwa/anaconda3/bin/activate
conda activate hclimai

set -exu 

cd $HOME/Script/HCLIMAI/src
python main.py -c config_main.ini 
#python srgans_fw.py

current_date_time="`date`";
echo The run ends at $current_date_time

exit 0 

