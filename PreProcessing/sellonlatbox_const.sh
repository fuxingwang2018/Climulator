#!/bin/bash
module load CDO/2.3.0-eccodes-aec-cmor-fftw-hpc2-intel-2023a-eb

EXP='FPS3'
GCM="ICHEC-EC-EARTH_RCP85_LC" #ECMWF-ERAINT
DOMAIN='Emilia_Romagna'

if [[ "$EXP" == "FPS12" ]]; then
    # https://en.wikipedia.org/wiki/Module:Location_map/data/Alps
    indir0='/nobackup/rossby26/proj/rossby/joint_exp/eucp/netcdf/HCLIM38-ALADIN/ALP-12/ECMWF-ERAINT/evaluation/'
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
    #indir0='/nobackup/rossby26/proj/rossby/joint_exp/eucp/CORDEX-FPSCONV/output/ALP-3/HCLIMcom/ECMWF-ERAINT/evaluation/r1i1p1/HCLIMcom-HCLIM38-AROME/fpsconv-x2yn2-v1/'
    #experiment='ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-HCLIM38-AROME_fpsconv-x2yn2-v1'
    # for fx fields like orog 
    if [[ "$GCM" == "ECMWF-ERAINT" ]]; then
        indir0='/nobackup/rossby26/proj/rossby/joint_exp/eucp/CORDEX-FPSCONV/output/ALP-3/HCLIMcom/ECMWF-ERAINT/evaluation/r0i0p0/HCLIMcom-HCLIM38-AROME/fpsconv-x2yn2-v1/'
        experiment='ALP-3_ECMWF-ERAINT_evaluation_r0i0p0_HCLIMcom-HCLIM38-AROME_fpsconv-x2yn2v1'
    elif [[ "$GCM" == "ICHEC-EC-EARTH" ]]; then
        indir0='/nobackup/rossby26/proj/rossby/joint_exp/eucp/CORDEX-FPSCONV/output/ALP-3/HCLIMcom/ICHEC-EC-EARTH/historical/r0i0p0/HCLIMcom-HCLIM38-AROME/fpsconv-x2yn2-v1/'
        experiment='ALP-3_ICHEC-EC-EARTH_historical_r0i0p0_HCLIMcom-HCLIM38-AROME_fpsconv-x2yn2v1'
    elif [[ "$GCM" == "ICHEC-EC-EARTH_RCP85_MC" ]] || [[ "$GCM" == "ICHEC-EC-EARTH_RCP85_LC" ]]; then
        indir0='/nobackup/rossby26/proj/rossby/joint_exp/eucp/CORDEX-FPSCONV/output/ALP-3/HCLIMcom/ICHEC-EC-EARTH/rcp85/r0i0p0/HCLIMcom-HCLIM38-AROME/fpsconv-x2yn2-v1/'
        experiment='ALP-3_ICHEC-EC-EARTH_rcp85_r0i0p0_HCLIMcom-HCLIM38-AROME_fpsconv-x2yn2v1'
    fi
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
    if [[ "$DOMAIN" == "Test_Domain" ]]; then
        OUT_PATH='/nobackup/rossby26/users/sm_fuxwa/AI/'$DOMAIN'/'$GCM/
        lonmin='9'
        lonmax='13'
        latmin='45.5'
        latmax='47.7'
    elif [[ "$DOMAIN" == "Emilia_Romagna" ]]; then
        OUT_PATH='/nobackup/rossby26/users/sm_fuxwa/AI/'$DOMAIN'/original/'$GCM/
        #ASPECT3 domain
        lonmin='4.0'
        lonmax='19.0'
        latmin='40.0'
        latmax='49.0'
    fi
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
