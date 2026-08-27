#!/bin/bash

#SBATCH -N 1
#SBATCH -t 3:00:00
#SBATCH -J ParDe
#SBATCH --chdir=/home/smf/log/log_ml
#SBATCH --error=%x-%j.error 
#SBATCH --output=%x-%j.out

module load cdo/2.5.3

#cdo selvar,albedo -seltime,2009-07-01,12:00:00 fileA.nc var_once.nc
#cdo timmean var_once.nc var_static.nc
#cdo addtime,var_static.nc fileB.nc var_expanded.nc
#cdo merge fileB.nc var_expanded.nc output.nc

VARNAME='tas'

INDIR1='/perm/smf/HCLIMAI/Emilia_Romagna/cropped/ECMWF-ERAINT/12km/6hr/'${VARNAME}'/'
INFILE1=${VARNAME}'_12km_6hr_200001010000-200912311800.nc'

INDIR2='/perm/smf/HCLIMAI/Emilia_Romagna/cropped/Emulator_HCLIM_CRM_T_withSM_whus/training_singleday/12km/6hr/'
INFILE2='JJA2003_20030815T1200_mrsol_whus_time.nc'

OUTDIR='/perm/smf/HCLIMAI/Emilia_Romagna/cropped/Emulator_HCLIM_CRM_T_withSMT_whus/training_singleday/12km/6hr/'
OUTFILE='JJA2003_20030815T1200_mrsol_whust_time.nc'


cdo merge ${INDIR2}/${INFILE2} \
  -duplicate,$(cdo -ntime ${INDIR2}/${INFILE2}) -timmean -selvar,${VARNAME} -seldate,2003-08-15 -seltime,12:00:00 ${INDIR1}/${INFILE1} \
  ${OUTDIR}/${OUTFILE}


