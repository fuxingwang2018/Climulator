#!/bin/bash
#SBATCH -N 1
#SBATCH -t 1:00:00
#SBATCH -J ParDe
#SBATCH --chdir=/nobackup/rossby26/users/sm_fuxwa/AI/log
#SBATCH --error=%x-%j.error 
#SBATCH --output=%x-%j.out
#SBATCH -A rossby

module load CDO/2.3.0-eccodes-aec-cmor-fftw-hpc2-intel-2023a-eb
module load NCO/5.1.3-hpc1-gcc-2022a-eb

VARNAME='mrsol' #'mrsol' #'tas'
RESOLUTION='12km' #'3km'
REMAP_METHOD='remapbil' #'remapnn' #'remapbil'

#DO_REMAP='FALSE' #'FALSE'
#DO_GRIDDES='TRUE'
#DO_SELTIME='FALSE' #'FALSE'
#DO_CREATEFORCING='FALSE'

DO_REMAP='TRUE' #'FALSE'
DO_GRIDDES='FALSE'
DO_SELTIME='TRUE' #'FALSE'
DO_CREATEFORCING='TRUE'

INDIR_GRIDDES='/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/cropped/ECMWF-ERAINT/'${RESOLUTION}'/6hr/'${VARNAME}'/'
INFILE_GRIDDES=${VARNAME}'_'${RESOLUTION}'_6hr_200001010000-200912311800.nc'

INDIR2='/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/cropped/Emulator_HCLIM_CRM_T_withSM_whus/training_singleday/12km/6hr/'
INFILE2='JJA2003_20030815T1200_mrsol_whus_time.nc'

#INDIR2='/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/cropped/Emulator_HCLIM_CRM_T_withSMT_whus/12km/6hr/'
#INFILE2='JJA2003_20030815T1200_mrsol_whust_time.nc'

if [[ "$DO_CREATEFORCING" == "TRUE" ]] ; then
    OUTDIR='/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/cropped/Emulator_HCLIM_CRM_T_withSM_whus_r1/'
    #OUTDIR='/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/cropped/Emulator_HCLIM_CRM_T_withSMT_whus_r1/'
else
    OUTDIR='/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/cropped/Emulator_HCLIM_Experiment_r1/'
fi

INDIR_BASE='/nobackup/rossby27/users/sm_aital/hm_home/HCLIM43_ASPECT_soil_moisture/V1_202608/'

if [[ "$VARNAME" == "mrsol" ]]; then
    FREQUENCY='6hr'
    SELTIMESTEP=3
    VARSUFFIX='L01_sfx'
elif [[ "$VARNAME" == "tas" ]]; then
    FREQUENCY='1hr'
    SELTIMESTEP=13
    VARSUFFIX='fp'
fi

if [[ "$RESOLUTION" == "12km" ]]; then
    EXPERIMENT='HCLIM43_ASPECT12_ALADIN'
    INFILENAME='ASPECT12_HCLIM43_ASPECT12_ALADIN'
    OUTFILENAME='HCLIM_ALADIN_12km'
elif [[ "$RESOLUTION" == "3km" ]]; then
    EXPERIMENT='HCLIM43_ASPECT3_AROME'
    INFILENAME='ASPECT3_HCLIM43_ASPECT3_AROME'
    OUTFILENAME='HCLIM_AROME_3km'
fi

if [ ! -e ${OUTDIR}/${REMAP_METHOD}/ ] ; then
    mkdir -p ${OUTDIR}/${REMAP_METHOD}/
elif [ ! -e ${OUTDIR}/processed_steps/ ] ; then
    mkdir -p ${OUTDIR}/processed_steps/
fi

FILE_025WWILT_OUT=${VARNAME}'_'${OUTFILENAME}'_025WWILT_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_050WWILT_OUT=${VARNAME}'_'${OUTFILENAME}'_050WWILT_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_075WWILT_OUT=${VARNAME}'_'${OUTFILENAME}'_075WWILT_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_WWILT_OUT=${VARNAME}'_'${OUTFILENAME}'_WWILT_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_WWILT_010_WFC_OUT=${VARNAME}'_'${OUTFILENAME}'_WWILT_010_WFC_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_WWILT_020_WFC_OUT=${VARNAME}'_'${OUTFILENAME}'_WWILT_020_WFC_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_WWILT_030_WFC_OUT=${VARNAME}'_'${OUTFILENAME}'_WWILT_030_WFC_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_WWILT_040_WFC_OUT=${VARNAME}'_'${OUTFILENAME}'_WWILT_040_WFC_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_WWILT_050_WFC_OUT=${VARNAME}'_'${OUTFILENAME}'_WWILT_050_WFC_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_WWILT_060_WFC_OUT=${VARNAME}'_'${OUTFILENAME}'_WWILT_060_WFC_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_WWILT_070_WFC_OUT=${VARNAME}'_'${OUTFILENAME}'_WWILT_070_WFC_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_WWILT_080_WFC_OUT=${VARNAME}'_'${OUTFILENAME}'_WWILT_080_WFC_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_WWILT_090_WFC_OUT=${VARNAME}'_'${OUTFILENAME}'_WWILT_090_WFC_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_WFC_OUT=${VARNAME}'_'${OUTFILENAME}'_WFC_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_WFC_025_WSAT_OUT=${VARNAME}'_'${OUTFILENAME}'_WFC_025_WSAT_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_WFC_050_WSAT_OUT=${VARNAME}'_'${OUTFILENAME}'_WFC_050_WSAT_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_WFC_075_WSAT_OUT=${VARNAME}'_'${OUTFILENAME}'_WFC_075_WSAT_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_WSAT_OUT=${VARNAME}'_'${OUTFILENAME}'_WSAT_'${FREQUENCY}'_200308150000-200308160000.nc'

if [[ "$DO_GRIDDES" == "TRUE" ]]; then
    cdo griddes ${INDIR_GRIDDES}/${INFILE_GRIDDES} > ${OUTDIR}/griddes_EmiliaRomagna_HCLIM_${RESOLUTION}.txt
    exit
fi

grid_target='/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/cropped/griddes_EmiliaRomagna_HCLIM_'${RESOLUTION}'.txt'
FILE_025WWILT_IN=${INDIR_BASE}'/'${EXPERIMENT}'_025WWILT/archive/2003/08/15/00/'${VARNAME}'_'${VARSUFFIX}'_'${INFILENAME}'_025WWILT_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_050WWILT_IN=${INDIR_BASE}'/'${EXPERIMENT}'_050WWILT/archive/2003/08/15/00/'${VARNAME}'_'${VARSUFFIX}'_'${INFILENAME}'_050WWILT_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_075WWILT_IN=${INDIR_BASE}'/'${EXPERIMENT}'_075WWILT/archive/2003/08/15/00/'${VARNAME}'_'${VARSUFFIX}'_'${INFILENAME}'_075WWILT_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_WWILT_IN=${INDIR_BASE}'/'${EXPERIMENT}'_WWILT/archive/2003/08/15/00/'${VARNAME}'_'${VARSUFFIX}'_'${INFILENAME}'_WWILT_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_WWILT_010_WFC_IN=${INDIR_BASE}'/'${EXPERIMENT}'_WWILT_010_WFC/archive/2003/08/15/00/'${VARNAME}'_'${VARSUFFIX}'_'${INFILENAME}'_WWILT_010_WFC_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_WWILT_020_WFC_IN=${INDIR_BASE}'/'${EXPERIMENT}'_WWILT_020_WFC/archive/2003/08/15/00/'${VARNAME}'_'${VARSUFFIX}'_'${INFILENAME}'_WWILT_020_WFC_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_WWILT_030_WFC_IN=${INDIR_BASE}'/'${EXPERIMENT}'_WWILT_030_WFC/archive/2003/08/15/00/'${VARNAME}'_'${VARSUFFIX}'_'${INFILENAME}'_WWILT_030_WFC_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_WWILT_040_WFC_IN=${INDIR_BASE}'/'${EXPERIMENT}'_WWILT_040_WFC/archive/2003/08/15/00/'${VARNAME}'_'${VARSUFFIX}'_'${INFILENAME}'_WWILT_040_WFC_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_WWILT_050_WFC_IN=${INDIR_BASE}'/'${EXPERIMENT}'_WWILT_050_WFC/archive/2003/08/15/00/'${VARNAME}'_'${VARSUFFIX}'_'${INFILENAME}'_WWILT_050_WFC_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_WWILT_060_WFC_IN=${INDIR_BASE}'/'${EXPERIMENT}'_WWILT_060_WFC/archive/2003/08/15/00/'${VARNAME}'_'${VARSUFFIX}'_'${INFILENAME}'_WWILT_060_WFC_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_WWILT_070_WFC_IN=${INDIR_BASE}'/'${EXPERIMENT}'_WWILT_070_WFC/archive/2003/08/15/00/'${VARNAME}'_'${VARSUFFIX}'_'${INFILENAME}'_WWILT_070_WFC_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_WWILT_080_WFC_IN=${INDIR_BASE}'/'${EXPERIMENT}'_WWILT_080_WFC/archive/2003/08/15/00/'${VARNAME}'_'${VARSUFFIX}'_'${INFILENAME}'_WWILT_080_WFC_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_WWILT_090_WFC_IN=${INDIR_BASE}'/'${EXPERIMENT}'_WWILT_090_WFC/archive/2003/08/15/00/'${VARNAME}'_'${VARSUFFIX}'_'${INFILENAME}'_WWILT_090_WFC_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_WFC_IN=${INDIR_BASE}'/'${EXPERIMENT}'_WFC/archive/2003/08/15/00/'${VARNAME}'_'${VARSUFFIX}'_'${INFILENAME}'_WFC_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_WFC_025_WSAT_IN=${INDIR_BASE}'/'${EXPERIMENT}'_WFC_025_WSAT/archive/2003/08/15/00/'${VARNAME}'_'${VARSUFFIX}'_'${INFILENAME}'_WFC_025_WSAT_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_WFC_050_WSAT_IN=${INDIR_BASE}'/'${EXPERIMENT}'_WFC_050_WSAT/archive/2003/08/15/00/'${VARNAME}'_'${VARSUFFIX}'_'${INFILENAME}'_WFC_050_WSAT_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_WFC_075_WSAT_IN=${INDIR_BASE}'/'${EXPERIMENT}'_WFC_075_WSAT/archive/2003/08/15/00/'${VARNAME}'_'${VARSUFFIX}'_'${INFILENAME}'_WFC_075_WSAT_'${FREQUENCY}'_200308150000-200308160000.nc'
FILE_WSAT_IN=${INDIR_BASE}'/'${EXPERIMENT}'_WSAT/archive/2003/08/15/00/'${VARNAME}'_'${VARSUFFIX}'_'${INFILENAME}'_WSAT_'${FREQUENCY}'_200308150000-200308160000.nc'


echo ${FILE_025WWILT_IN}
echo ${OUTDIR}/${REMAP_METHOD}/${FILE_025WWILT_OUT}

if [[ "$DO_REMAP" == "TRUE" ]]; then
	cdo ${REMAP_METHOD},${grid_target} ${FILE_025WWILT_IN} ${OUTDIR}/${REMAP_METHOD}/${FILE_025WWILT_OUT}
	cdo ${REMAP_METHOD},${grid_target} ${FILE_050WWILT_IN} ${OUTDIR}/${REMAP_METHOD}/${FILE_050WWILT_OUT}
	cdo ${REMAP_METHOD},${grid_target} ${FILE_075WWILT_IN} ${OUTDIR}/${REMAP_METHOD}/${FILE_075WWILT_OUT}
	cdo ${REMAP_METHOD},${grid_target} ${FILE_WWILT_IN} ${OUTDIR}/${REMAP_METHOD}/${FILE_WWILT_OUT}
	cdo ${REMAP_METHOD},${grid_target} ${FILE_WWILT_010_WFC_IN} ${OUTDIR}/${REMAP_METHOD}/${FILE_WWILT_010_WFC_OUT}
	cdo ${REMAP_METHOD},${grid_target} ${FILE_WWILT_020_WFC_IN} ${OUTDIR}/${REMAP_METHOD}/${FILE_WWILT_020_WFC_OUT}
	cdo ${REMAP_METHOD},${grid_target} ${FILE_WWILT_030_WFC_IN} ${OUTDIR}/${REMAP_METHOD}/${FILE_WWILT_030_WFC_OUT}
	cdo ${REMAP_METHOD},${grid_target} ${FILE_WWILT_040_WFC_IN} ${OUTDIR}/${REMAP_METHOD}/${FILE_WWILT_040_WFC_OUT}
	cdo ${REMAP_METHOD},${grid_target} ${FILE_WWILT_050_WFC_IN} ${OUTDIR}/${REMAP_METHOD}/${FILE_WWILT_050_WFC_OUT}
	cdo ${REMAP_METHOD},${grid_target} ${FILE_WWILT_060_WFC_IN} ${OUTDIR}/${REMAP_METHOD}/${FILE_WWILT_060_WFC_OUT}
	cdo ${REMAP_METHOD},${grid_target} ${FILE_WWILT_070_WFC_IN} ${OUTDIR}/${REMAP_METHOD}/${FILE_WWILT_070_WFC_OUT}
	cdo ${REMAP_METHOD},${grid_target} ${FILE_WWILT_080_WFC_IN} ${OUTDIR}/${REMAP_METHOD}/${FILE_WWILT_080_WFC_OUT}
	cdo ${REMAP_METHOD},${grid_target} ${FILE_WWILT_090_WFC_IN} ${OUTDIR}/${REMAP_METHOD}/${FILE_WWILT_090_WFC_OUT}
	cdo ${REMAP_METHOD},${grid_target} ${FILE_WFC_IN} ${OUTDIR}/${REMAP_METHOD}/${FILE_WFC_OUT}
	cdo ${REMAP_METHOD},${grid_target} ${FILE_WFC_025_WSAT_IN} ${OUTDIR}/${REMAP_METHOD}/${FILE_WFC_025_WSAT_OUT}
	cdo ${REMAP_METHOD},${grid_target} ${FILE_WFC_050_WSAT_IN} ${OUTDIR}/${REMAP_METHOD}/${FILE_WFC_050_WSAT_OUT}
	cdo ${REMAP_METHOD},${grid_target} ${FILE_WFC_075_WSAT_IN} ${OUTDIR}/${REMAP_METHOD}/${FILE_WFC_075_WSAT_OUT}
	cdo ${REMAP_METHOD},${grid_target} ${FILE_WSAT_IN} ${OUTDIR}/${REMAP_METHOD}/${FILE_WSAT_OUT}
fi

if [[ "$DO_SELTIME" == "TRUE" ]]; then
	for file in ${OUTDIR}/${REMAP_METHOD}/${VARNAME}_${OUTFILENAME}*.nc; do
	    base=$(basename "$file")
	    
	    # Extract 3rd timestep -> remap to target grid -> rename mrsol_L01 to mrsol
            if [[ "$VARNAME" == "mrsol" ]]; then
	        cdo -chname,mrsol_L01,mrsol \
		    -seltimestep,${SELTIMESTEP} \
		    "$file" ${OUTDIR}"/processed_steps/"${base}
            elif [[ "$VARNAME" == "tas" ]]; then
	        cdo -seltimestep,${SELTIMESTEP} \
		    "$file" ${OUTDIR}"/processed_steps/"${base}
            fi
	done
fi

# Concatenate the 18 processed steps into a single file
cdo mergetime ${OUTDIR}"/processed_steps/"${VARNAME}"_"${OUTFILENAME}_"*200308150000-200308160000.nc" ${OUTDIR}"/processed_steps/"${VARNAME}"_"${OUTFILENAME}"_replacement_all_18_steps.nc"
cdo setmisstoc,0 ${OUTDIR}"/processed_steps/"${VARNAME}"_"${OUTFILENAME}"_replacement_all_18_steps.nc" ${OUTDIR}"/processed_steps/"${VARNAME}"_"${OUTFILENAME}"_missing0_replacement_all_18_steps.nc"

if [[ "$DO_CREATEFORCING" == "TRUE" ]] ; then

    # Slice original target file to 18 steps
    cdo seltimestep,1/18 ${INDIR2}/${INFILE2} ${OUTDIR}"processed_steps/"${VARNAME}"_"${OUTFILENAME}"_template_all_18_steps.nc"

    # Overwrite the mrsol variable values 
    #ncks -x -v _FillValue -A -v ${VARNAME} ${OUTDIR}"processed_steps/"${VARNAME}"_"${OUTFILENAME}"_replacement_all_18_steps.nc" \
    #    ${OUTDIR}"processed_steps/"${VARNAME}"_"${OUTFILENAME}"_template_all_18_steps.nc"

    cdo -delname,${VARNAME} ${OUTDIR}"processed_steps/"${VARNAME}"_"${OUTFILENAME}"_template_all_18_steps.nc" \
        ${OUTDIR}"processed_steps/"${VARNAME}"_deleted_"${OUTFILENAME}"_template_all_18_steps.nc" 

    cdo merge ${OUTDIR}"processed_steps/"${VARNAME}"_deleted_"${OUTFILENAME}"_template_all_18_steps.nc" \
        ${OUTDIR}"processed_steps/"${VARNAME}"_"${OUTFILENAME}"_missing0_replacement_all_18_steps.nc" \
        ${OUTDIR}"processed_steps/"${VARNAME}"_replaced_"${OUTFILENAME}"_replacement_all_18_steps.nc"

    # Correct time axis
    cdo settaxis,2003-06-01,00:00:00,6hour \
        ${OUTDIR}"processed_steps/"${VARNAME}"_replaced_"${OUTFILENAME}"_replacement_all_18_steps.nc" \
        ${OUTDIR}"processed_steps/"${VARNAME}"_replaced_"${OUTFILENAME}"_replacement_all_18_steps_time_updated.nc" 


    # Rename output file
    #mv ${OUTDIR}"processed_steps/template_all_18_steps_time_updated.nc" ${OUTDIR}"/"${RESOLUTION}"/6hr/JJA2003_20030815T1200_mrsol_whus_wsm_"${RESOLUTION}"_6hr.nc"
    #mv ${OUTDIR}"processed_steps/"${VARNAME}"_"${OUTFILENAME}"_template_all_18_steps_time_updated.nc" \
    mv ${OUTDIR}"processed_steps/"${VARNAME}"_replaced_"${OUTFILENAME}"_replacement_all_18_steps_time_updated.nc" \
        ${OUTDIR}"/"${RESOLUTION}"/6hr/JJA2003_20030815T1200_"${VARNAME}"_whus_wsm_wt_"${RESOLUTION}"_6hr.nc"

else
    # Correct time axis
    cdo settaxis,2003-06-01,00:00:00,6hour \
        ${OUTDIR}"processed_steps/"${VARNAME}"_"${OUTFILENAME}"_replacement_all_18_steps.nc" \
        ${OUTDIR}"processed_steps/"${VARNAME}"_"${OUTFILENAME}"_replacement_all_18_steps_time_updated.nc"

    mv ${OUTDIR}"processed_steps/"${VARNAME}"_"${OUTFILENAME}"_replacement_all_18_steps_time_updated.nc" \
        ${OUTDIR}"/"${RESOLUTION}"/6hr/"${VARNAME}"_"${RESOLUTION}"_PDP_"${FREQUENCY}"_200306010000-200306050600.nc"
    #rm -f ${OUTDIR}/${REMAP_METHOD}/*
    #rm -f ${OUTDIR}/processed_steps/*
fi

