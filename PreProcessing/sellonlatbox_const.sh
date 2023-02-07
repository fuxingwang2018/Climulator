#!/bin/bash

EXP='NorCP12'

if [[ "$EXP" == "FPS12" ]]; then
    # https://en.wikipedia.org/wiki/Module:Location_map/data/Alps
    indir0='/nobackup/rossby25/proj/rossby/joint_exp/eucp/netcdf/HCLIM38-ALADIN/ALP-12/ECMWF-ERAINT/evaluation/'
    experiment='ALP-12_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-HCLIM38-ALADIN_v1'

    #indir0='/nobackup/rossby26/users/sm_fuxwa/AI/CORDEX_FPS_ALP12_ERAI_CMORise'
    #experiment='ALP-12_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-SMHI-HCLIM38-ALADIN_v1'

    experiment_cmorized='12km'

    #lonmin='4.5'
    #lonmax='17'
    #latmin='42.75'
    #latmax='48.5'
    freq_in='fx' 
    freq_out='fx' 
    VAR_LIST=('orog') 

elif [[ "$EXP" == "NorCP12" ]]; then

    indir0='/nobackup/rossby24/proj/rossby/joint_exp/norcp/netcdf/NorCP_ALADIN_ECE_1985_2005/'
    experiment='NEU-12_ICHEC-EC-EARTH_historical_r12i1p1_HCLIMcom-HCLIM38-ALADIN_v1'

    experiment_cmorized='12km'
    freq_in='fx' 
    freq_out='fx' 
    VAR_LIST=('orog') 

elif [[ "$EXP" == "FPS3" ]]; then 
    indir0='/nobackup/rossby25/proj/rossby/joint_exp/eucp/CORDEX-FPSCONV/output/ALP-3/HCLIMcom/ECMWF-ERAINT/evaluation/r1i1p1/HCLIMcom-HCLIM38-AROME/fpsconv-x2yn2-v1/'
    experiment='ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-HCLIM38-AROME_fpsconv-x2yn2-v1' 
    experiment_cmorized='3km'
    freq_in='fx'
    freq_out='fx'
    VAR_LIST=('orog')  
    #lonmin='9'
    #lonmax='13'
    #latmin='45.5'
    #latmax='47.7'
fi


if [[ "$EXP" == "NorCP12" || "$EXP" == "NorCP3" ]]; then
    OUT_PATH='/nobackup/rossby26/users/sm_fuxwa/AI/NorCP/'
    lonmin='13.0'
    lonmax='16.3'
    latmin='57.1'
    latmax='59.5'
else
    OUT_PATH='/nobackup/rossby26/users/sm_fuxwa/AI/'
    lonmin='9'
    lonmax='13'
    latmin='45.5'
    latmax='47.7'
fi

for ivar in ${VAR_LIST[@]} ; do
    indir=${indir0}/${freq_in}/${ivar}
    outdir=${OUT_PATH}${experiment_cmorized}/${freq_out}/${ivar}
    if [ ! -e ${outdir} ] ; then
        mkdir -p ${outdir}
    fi
    infile=${ivar}_${experiment}_${freq_in}'.nc' 
    outfile_smalldomain=${ivar}_${experiment}_${freq_in}'_smalldomain.nc'
    outfile_cmorized=${ivar}_${experiment_cmorized}_${freq_out}'.nc'

    cdo sellonlatbox,$lonmin,$lonmax,$latmin,$latmax $indir/$infile $outdir/$outfile_smalldomain
    mv $outdir/$outfile_smalldomain  $outdir/$outfile_cmorized
done
