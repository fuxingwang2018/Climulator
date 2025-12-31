#!/bin/bash

#SBATCH -N 1
#SBATCH -t 3:00:00
#SBATCH -J timeaxis
#SBATCH --chdir=/home/smf/log/log_ml
#SBATCH --error=%x-%j.error 
#SBATCH --output=%x-%j.out

module load cdo/2.5.3

VARNAME='tas'
EPOCH1_tas_wsm_scale_time_bs50_val0.1_atos

INDIR='/ec/res4/scratch/smf/HCLIMAI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsm_scale_time_bs50_val0.1_atos/'
INFILE='predictor_1_timewrong.nc'
#INFILE='predictant_ypred_1_timewrong.nc'
#INFILE='predictant_ytest_1_timewrong.nc'

OUTDIR=${INDIR}
OUTFILE='predictor_1.nc'
#OUTFILE='predictant_ypred_1.nc'
#OUTFILE='predictant_ytest_1.nc'


cdo settaxis,2009-01-01,00:00:00,6hour -seltimestep,1/1460 ${INDIR}/${INFILE} ${OUTDIR}/${OUTFILE}
