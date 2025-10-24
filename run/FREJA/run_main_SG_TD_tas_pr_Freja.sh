#!/bin/bash 
#SBATCH -N 1 
###SBATCH -t 01:00:00 
###SBATCH -t 16:00:00 
#SBATCH -t 40:00:00 
###SBATCH -n 1  ##ntasks 
###SBATCH --mem=256000 
#SBATCH -J SGTDtaspr 
#SBATCH --chdir=/nobackup/rossby26/users/sm_fuxwa/AI/log
#SBATCH --error=%x-%j.error 
#SBATCH --output=%x-%j.out
#SBATCH -A rossby
###SBATCH --qos=low

module load netCDF-HDF5-utils/4.9.2-1.12.2-hpc1-intel-2023a-eb

#DOMAIN='EmiliaRomagna'
DOMAIN='TestDomain'
VARIABLE='tas_pr'

current_date_time="`date`";
echo The run starts from $current_date_time

#source ~sm_fuxwa/anaconda3/bin/activate
source /nobackup/rossby24/users/sm_fuxwa/conda/miniconda3/bin/activate
conda activate hclimai

set -exu 

cd $HOME/Script/Climulator/src
python main.py -c ../config/config_main_SG_${DOMAIN}_${VARIABLE}_Freja.ini 
#python srgans_fw.py

#cd $HOME/Script/Climulator
#python -m pytest

current_date_time="`date`";
echo The run ends at $current_date_time

exit 0 

