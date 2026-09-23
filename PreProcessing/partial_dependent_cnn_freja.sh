
#!/bin/bash
#SBATCH -N 1
#SBATCH -t 1:00:00
#SBATCH -J ParDe
#SBATCH --chdir=/nobackup/rossby26/users/sm_fuxwa/AI/log
#SBATCH --error=%x-%j.error 
#SBATCH --output=%x-%j.out
#SBATCH -A rossby

#INDIR1='/nobackup/rossby27/users/sm_yicwa/DATA_shared/Climulator/Emulator_HCLIM_CRM_T_withSM_whus/'
INDIR1='/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/cropped/Emulator_HCLIM_Experiment_r1/CNN/'
INFILE1='simple_cnn_prediction_normalized_20030815T1200.nc'

#INDIR2='/nobackup/rossby27/users/sm_yicwa/DATA_shared/AIES_revision_aug2026/TAS_pdp_r1/'
INDIR2='/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/cropped/Emulator_HCLIM_Experiment_r1/CNN/'
INFILE2='simple_cnn_prediction_normalized_20030815T1200_r1_only20.nc'
VARNAME='tas'

OUTDIR='/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/cropped/Emulator_HCLIM_Experiment_r1/CNN/'
OUTFILE_1step='simple_cnn_prediction_normalized_20030815T1200_only_1_step.nc'
OUTFILE_18step='simple_cnn_prediction_normalized_20030815T1200_r1_18_steps.nc'

OUTFILE='simple_cnn_prediction_normalized_20030815T1200_r1_19_steps.nc'

VARNAME='test'

if [ ! -e ${OUTDIR}/processed_steps/ ] ; then
    mkdir -p ${OUTDIR}/processed_steps/
else
    rm -fr ${OUTDIR}/processed_steps/*.nc
fi

if [ -e ${OUTDIR}"/"${OUTFILE} ] ; then
    rm -fr ${OUTDIR}/${OUTFILE}
fi

#-selname,${VARNAME} \
cdo -seltimestep,303 \
    ${INDIR1}/${INFILE1} ${OUTDIR}"/processed_steps/"${OUTFILE_1step}

cdo -seltimestep,1/18 \
    ${INDIR2}/${INFILE2} ${OUTDIR}"/processed_steps/"${OUTFILE_18step}

files_to_merge=(\
    ${OUTDIR}"/processed_steps/"${OUTFILE_1step} \
    ${OUTDIR}"/processed_steps/"${OUTFILE_18step} 
)
cdo cat \
    "${files_to_merge[@]}" \
    ${OUTDIR}"/"${OUTFILE}

