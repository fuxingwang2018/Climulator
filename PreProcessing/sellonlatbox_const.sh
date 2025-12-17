#!/bin/bash 
#SBATCH -N 1 
#SBATCH -t 01:00:00 
###SBATCH -n 1  ##ntasks 
###SBATCH --mem=16G
#SBATCH -J prep 
#SBATCH --chdir=/nobackup/rossby26/users/sm_fuxwa/AI/log_stats
###SBATCH --chdir=/nobackup/rossby27/users/sm_yicwa/PROJECTS/01-PROJ_emulator/04-evaluation_fuxing/ClimulatorScore
#SBATCH --error=%x-%j.error 
#SBATCH --output=%x-%j.out
###SBATCH --ntasks=1
#SBATCH -A rossby
###SBATCH --qos=low

module load CDO/2.3.0-eccodes-aec-cmor-fftw-hpc2-intel-2023a-eb

EXP='NorCP3'
GCM="ECMWF-ERAINT"
#GCM="ICHEC-EC-EARTH_HIST" 
#GCM="ICHEC-EC-EARTH_RCP85_MC"
#GCM="ICHEC-EC-EARTH_RCP85_LC"
#GCM="ICHEC-EC-EARTH_RCP45_MC"
#GCM="ICHEC-EC-EARTH_RCP45_LC"
#DOMAIN='Emilia_Romagna'
DOMAIN='NorCP_SSE'

freq_in='fx'
freq_out='fx'
VAR_LIST=('orog')  

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

elif [[ "$EXP" == "NorCP12" ]]; then
    if [[ "$GCM" == "ECMWF-ERAINT" ]]; then
        indir0='/nobackup/rossby24/proj/rossby/joint_exp/norcp/netcdf/NorCP_ALADIN_ERAI_1997_2018/'
        experiment='NEU-12_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-HCLIM38-ALADIN_v1'
    elif [[ "$GCM" == "ICHEC-EC-EARTH_HIST" ]]; then
        indir0='/nobackup/rossby24/proj/rossby/joint_exp/norcp/netcdf/NorCP_ALADIN_ECE_1985_2005/'
        experiment='NEU-12_ICHEC-EC-EARTH_historical_r12i1p1_HCLIMcom-HCLIM38-ALADIN_v1'
    elif [[ "$GCM" == "ICHEC-EC-EARTH_RCP85_LC" ]]; then
        indir0='/nobackup/rossby24/proj/rossby/joint_exp/norcp/netcdf/NorCP_ALADIN_ECE_RCP85_2080_2100/'
        experiment='NEU-12_ICHEC-EC-EARTH_rcp85_r12i1p1_HCLIMcom-HCLIM38-ALADIN_v1'
    elif [[ "$GCM" == "ICHEC-EC-EARTH_RCP85_MC" ]]; then
        indir0='/nobackup/rossby24/proj/rossby/joint_exp/norcp/netcdf/NorCP_ALADIN_ECE_RCP85_2040_2060/'
        experiment='NEU-12_ICHEC-EC-EARTH_rcp85_r12i1p1_HCLIMcom-HCLIM38-ALADIN_v1'
    elif [[ "$GCM" == "ICHEC-EC-EARTH_RCP45_LC" ]]; then
        indir0='/nobackup/rossby24/proj/rossby/joint_exp/norcp/netcdf/NorCP_ALADIN_ECE_RCP45_2080_2100/'
        experiment='NEU-12_ICHEC-EC-EARTH_rcp45_r12i1p1_HCLIMcom-HCLIM38-ALADIN_v1'
    elif [[ "$GCM" == "ICHEC-EC-EARTH_RCP45_MC" ]]; then
        indir0='/nobackup/rossby24/proj/rossby/joint_exp/norcp/netcdf/NorCP_ALADIN_ECE_RCP45_2040_2060/'
        experiment='NEU-12_ICHEC-EC-EARTH_rcp45_r12i1p1_HCLIMcom-HCLIM38-ALADIN_v1'
    fi


    experiment_cmorized='12km'

elif [[ "$EXP" == "NorCP3" ]]; then
    if [[ "$GCM" == "ECMWF-ERAINT" ]]; then
        indir0='/nobackup/rossby24/proj/rossby/joint_exp/norcp/netcdf/NorCP_AROME_ERAI_ALADIN_1998_2018/'
        experiment='NEU-3_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-HCLIM38-AROME_x2yn2v1'
    elif [[ "$GCM" == "ICHEC-EC-EARTH_HIST" ]]; then
        indir0='/nobackup/rossby24/proj/rossby/joint_exp/norcp/netcdf/NorCP_AROME_ECE_ALADIN_1985_2005/'
        experiment='NEU-3_ICHEC-EC-EARTH_historical_r12i1p1_HCLIMcom-HCLIM38-AROME_x2yn2v1' 
    elif [[ "$GCM" == "ICHEC-EC-EARTH_RCP85_LC" ]]; then
        indir0='/nobackup/rossby24/proj/rossby/joint_exp/norcp/netcdf/NorCP_AROME_ECE_ALADIN_RCP85_2080_2100/'
        experiment='NEU-3_ICHEC-EC-EARTH_rcp85_r12i1p1_HCLIMcom-HCLIM38-AROME_x2yn2v1'
    elif [[ "$GCM" == "ICHEC-EC-EARTH_RCP85_MC" ]]; then
        indir0='/nobackup/rossby24/proj/rossby/joint_exp/norcp/netcdf/NorCP_AROME_ECE_ALADIN_RCP85_2040_2060/'
        experiment='NEU-3_ICHEC-EC-EARTH_rcp85_r12i1p1_HCLIMcom-HCLIM38-AROME_x2yn2v1'
    elif [[ "$GCM" == "ICHEC-EC-EARTH_RCP45_LC" ]]; then
        indir0='/nobackup/rossby24/proj/rossby/joint_exp/norcp/netcdf/NorCP_AROME_ECE_ALADIN_RCP45_2080_2100/'
        experiment='NEU-3_ICHEC-EC-EARTH_rcp45_r12i1p1_HCLIMcom-HCLIM38-AROME_x2yn2v1'
    elif [[ "$GCM" == "ICHEC-EC-EARTH_RCP45_MC" ]]; then
        indir0='/nobackup/rossby24/proj/rossby/joint_exp/norcp/netcdf/NorCP_AROME_ECE_ALADIN_RCP45_2040_2060/'
        experiment='NEU-3_ICHEC-EC-EARTH_rcp45_r12i1p1_HCLIMcom-HCLIM38-AROME_x2yn2v1'
    fi


    experiment_cmorized='3km'

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
    #lonmin='9'
    #lonmax='13'
    #latmin='45.5'
    #latmax='47.7'
fi


if [[ "$EXP" == "NorCP12" || "$EXP" == "NorCP3" ]]; then
    OUT_PATH='/nobackup/rossby26/users/sm_fuxwa/AI/'$DOMAIN'/original/'${GCM}/
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

    echo 'infile:', $indir/$infile
    echo 'outfile:', $outdir/$outfile_smalldomain

    cdo sellonlatbox,$lonmin,$lonmax,$latmin,$latmax $indir/$infile $outdir/$outfile_smalldomain
    mv $outdir/$outfile_smalldomain  $outdir/$outfile_cmorized
done
