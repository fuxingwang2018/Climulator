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

EXP='FPS3' #FPS12
FIRST_MONTH=01
LAST_MONTH=12
#DOMAIN='Test_Domain' #Emilia_Romagna
DOMAIN='Emilia_Romagna'
#DOMAIN='ALP'
#GCM="ECMWF-ERAINT" #"ICHEC-EC-EARTH_RCP85_MC" #ECMWF-ERAINT
#GCM="ICHEC-EC-EARTH_HIST" #, 
GCM="ICHEC-EC-EARTH_RCP85_MC"

if [[ "$EXP" == "FPS12" ]]; then
    # https://en.wikipedia.org/wiki/Module:Location_map/data/Alps
    # for tas, pr, CAPE  clivi  clt  clwvi  huss  prra  prsn  prw  ps  uas  vas (1hr)
    # for hfls  hfss  hus500  mrfso  mrso  rlds  rlns  rsds  rsns  ta500  va850 (3hr)
    if [[ "$GCM" == "ECMWF-ERAINT" ]]; then
        indir0='/nobackup/rossby26/proj/rossby/joint_exp/eucp/netcdf/HCLIM38-ALADIN/ALP-12/ECMWF-ERAINT/evaluation/'
        experiment='ALP-12_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-HCLIM38-ALADIN_v1'
        FIRST_YEAR=2000
        LAST_YEAR=2009 #2014 available but 2009 to consistent with FPS3, 2009 discarded because of spinup
    elif [[ "$GCM" == "ICHEC-EC-EARTH_HIST" ]]; then
        indir0='/nobackup/rossby26/proj/rossby/joint_exp/eucp/netcdf/HCLIM38-ALADIN/ALP-12/ICHEC-EC-EARTH/historical/'
        experiment='ALP-12_ICHEC-EC-EARTH_historical_r12i1p1_HCLIMcom-HCLIM38-ALADIN_v1'
        FIRST_YEAR=1995
        LAST_YEAR=2005 #2014 available but 2009 to consistent with FPS3, 2009 discarded because of spinup
    elif [[ "$GCM" == "ICHEC-EC-EARTH_RCP85_MC" ]]; then
        indir0='/nobackup/rossby26/proj/rossby/joint_exp/eucp/netcdf/HCLIM38-ALADIN/ALP-12/ICHEC-EC-EARTH/rcp85/'
        experiment='ALP-12_ICHEC-EC-EARTH_rcp85_r12i1p1_HCLIMcom-HCLIM38-ALADIN_v1'
        FIRST_YEAR=2040
        LAST_YEAR=2050 
    elif [[ "$GCM" == "ICHEC-EC-EARTH_RCP85_LC" ]]; then
        indir0='/nobackup/rossby26/proj/rossby/joint_exp/eucp/netcdf/HCLIM38-ALADIN/ALP-12/ICHEC-EC-EARTH/rcp85/'
        experiment='ALP-12_ICHEC-EC-EARTH_rcp85_r12i1p1_HCLIMcom-HCLIM38-ALADIN_v1'
        FIRST_YEAR=2089
        LAST_YEAR=2099 
    fi

    # for mrsol, ta500..950, hus500..950, ua500..950, va500..950 (3hr), phi500..950 (6hr)
    # snc, snw, snd (3hr)
    if [[ "$GCM" == "ECMWF-ERAINT" ]]; then
        indir0='/nobackup/rossby26/users/sm_fuxwa/AI/CORDEX_FPS_ALP12_ERAI_CMORise' 
        #same as /nobackup/rossby25/proj/rossby/joint_exp/eucp/netcdf/HCLIM38-ALADIN/ALP-12/ECMWF-ERAINT/evaluation/ but different variables
        experiment='ALP-12_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-SMHI-HCLIM38-ALADIN_v1'
    elif [[ "$GCM" == "ICHEC-EC-EARTH_HIST" ]]; then
        indir0='/nobackup/rossby26/users/sm_fuxwa/AI/CORDEX_FPS_ALP12_ECEARTH_HIST_CMORise' 
        experiment='ALP-12_ICHEC-EC-EARTH_historical_r12i1p1_HCLIMcom-METNo-HCLIM38-ALADIN_v1'
    elif [[ "$GCM" == "ICHEC-EC-EARTH_RCP85_MC" ]]; then
        indir0='/nobackup/rossby26/users/sm_fuxwa/AI/CORDEX_FPS_ALP12_ECEARTH_RCP85_MC_CMORise' 
        experiment='ALP-12_ICHEC-EC-EARTH_rcp85_r12i1p1_HCLIMcom-SMHI-HCLIM38-ALADIN_v1'
    elif [[ "$GCM" == "ICHEC-EC-EARTH_RCP85_LC" ]]; then
        indir0='/nobackup/rossby26/users/sm_fuxwa/AI/CORDEX_FPS_ALP12_ECEARTH_RCP85_LC_CMORise' 
        experiment='ALP-12_ICHEC-EC-EARTH_rcp85_r12i1p1_HCLIMcom-METNo-HCLIM38-ALADIN_v1'
    fi

    experiment_cmorized='12km'
    #lonmin='4.5'
    #lonmax='17'
    #latmin='42.75'
    #latmax='48.5'
    freq_in='3hr' #'day' #'3hr' #1hr, 3hr, 6hr, fx
    freq_out='day' #'6hr' #fx
    ## VAR_LIST=('orog' ) # orog, does not work, use sellonlatbox_const.sh
    #VAR_LIST=('tas' ) # pr, tas
    #VAR_LIST=('mrso' 'mrsol' ) # pr, tas
    #VAR_LIST=('mrsol' ) # pr, tas
    #VAR_LIST=('hus500' )
    #VAR_LIST=('CAPE' 'clt' 'huss' 'ps' 'uas' 'vas' ) 
    #VAR_LIST=('hfls'  'hfss'  'mrfso'  'mrso'  'mrsol'	'rlds'  'rlns'  'rsds'  'rsns')
    #VAR_LIST=('ta500' 'ta700' 'ta850' 'ta950' \
    #	'hus500' 'hus700' 'hus850' 'hus950' \
    #	'ua500' 'ua700' 'ua850' 'ua950' \
    #	'va500' 'va700' 'va850' 'va950') # 3hr
    #VAR_LIST=('phi500' 'phi700' 'phi850' 'phi950') # 6hr
    #VAR_LIST=('snc')
    VAR_LIST=('snd' 'snw')

elif [[ "$EXP" == "FPS3" ]]; then 
    if [[ "$GCM" == "ECMWF-ERAINT" ]]; then
        indir0='/nobackup/rossby26/proj/rossby/joint_exp/eucp/CORDEX-FPSCONV/output/ALP-3/HCLIMcom/ECMWF-ERAINT/evaluation/r1i1p1/HCLIMcom-HCLIM38-AROME/fpsconv-x2yn2-v1/'
        experiment='ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-HCLIM38-AROME_fpsconv-x2yn2-v1' 
        FIRST_YEAR=2000 # 1999 available but we discard it for spinup
        LAST_YEAR=2009
    elif [[ "$GCM" == "ICHEC-EC-EARTH_HIST" ]]; then
        indir0='/nobackup/rossby26/proj/rossby/joint_exp/eucp/CORDEX-FPSCONV/output/ALP-3/HCLIMcom/ICHEC-EC-EARTH/historical/r12i1p1/HCLIMcom-HCLIM38-AROME/fpsconv-x2yn2-v1/'
        experiment='ALP-3_ICHEC-EC-EARTH_historical_r12i1p1_HCLIMcom-HCLIM38-AROME_fpsconv-x2yn2-v1'
        FIRST_YEAR=1995 # 1999 available but we discard it for spinup
        LAST_YEAR=2005
    elif [[ "$GCM" == "ICHEC-EC-EARTH_RCP85_MC" ]]; then
        indir0='/nobackup/rossby26/proj/rossby/joint_exp/eucp/CORDEX-FPSCONV/output/ALP-3/HCLIMcom/ICHEC-EC-EARTH/rcp85/r12i1p1/HCLIMcom-HCLIM38-AROME/fpsconv-x2yn2-v1/'
        experiment='ALP-3_ICHEC-EC-EARTH_rcp85_r12i1p1_HCLIMcom-HCLIM38-AROME_fpsconv-x2yn2-v1'
        FIRST_YEAR=2040 
        LAST_YEAR=2050
    elif [[ "$GCM" == "ICHEC-EC-EARTH_RCP85_LC" ]]; then
        indir0='/nobackup/rossby26/proj/rossby/joint_exp/eucp/CORDEX-FPSCONV/output/ALP-3/HCLIMcom/ICHEC-EC-EARTH/rcp85/r12i1p1/HCLIMcom-HCLIM38-AROME/fpsconv-x2yn2-v1/'
        experiment='ALP-3_ICHEC-EC-EARTH_rcp85_r12i1p1_HCLIMcom-HCLIM38-AROME_fpsconv-x2yn2-v1'
        FIRST_YEAR=2089
        LAST_YEAR=2099
    fi

    if [[ "$GCM" == "ECMWF-ERAINT" ]]; then
        indir0=''
        experiment='ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-HCLIM38-AROME_fpsconv-x2yn2-v1' 
        FIRST_YEAR=2000 # 1999 available but we discard it for spinup
        LAST_YEAR=2009
    elif [[ "$GCM" == "ICHEC-EC-EARTH_HIST" ]]; then
        indir0='/nobackup/rossby26/users/sm_fuxwa/AI/CORDEX_FPS_ALP3_ECEARTH_HIST_CMORise/'
        experiment='ALP-3_ICHEC-EC-EARTH_historical_r12i1p1_HCLIMcom-HCLIM38-AROME_fpsconv-x2yn2-v1'
        FIRST_YEAR=1995 
        LAST_YEAR=2005
    elif [[ "$GCM" == "ICHEC-EC-EARTH_RCP85_MC" ]]; then
        indir0='//nobackup/rossby26/users/sm_fuxwa/AI/CORDEX_FPS_ALP3_ECEARTH_RCP85_MC_CMORise/'
        #experiment='ALP-3_ICHEC-EC-EARTH_rcp85_r12i1p1_HCLIMcom-HCLIM38-AROME_fpsconv-x2yn2-v1'
        experiment='ALP-3_ICHEC-EC-EARTH_rcp85_r12i1p1_HCLIMcom-SMHI-HCLIM38-AROME_v1'
        FIRST_YEAR=2040 
        LAST_YEAR=2050
    elif [[ "$GCM" == "ICHEC-EC-EARTH_RCP85_LC" ]]; then
        indir0=''
        experiment='ALP-3_ICHEC-EC-EARTH_rcp85_r12i1p1_HCLIMcom-HCLIM38-AROME_fpsconv-x2yn2-v1'
        FIRST_YEAR=2089
        LAST_YEAR=2099
    fi

    experiment_cmorized='3km'
    freq_in='3hr' #'1hr'  # '3hr' for mrsol
    freq_out='day' #'6hr'
    #VAR_LIST=('tas') #('mrsol' 'mrso')  # mrsol, mrso, hfls, tas, pr
    #VAR_LIST=('snc' 'snd' 'snw') # 6hr
    VAR_LIST=('snw') # 6hr
fi

SELMONTH=7
NAMEMONTH='July'

if [[ "$DOMAIN" == "Test_Domain" ]]; then

    # Smaller test domain
    lonmin='9'
    lonmax='13'
    latmin='45.5'
    latmax='47.7'
    OUT_PATH='/nobackup/rossby26/users/sm_fuxwa/AI/'$DOMAIN'/'$GCM/

elif [[ "$DOMAIN" == "Emilia_Romagna" ]]; then

    #ASPECT3 domain
    lonmin='4.0'
    lonmax='19.0'
    latmin='40.0'
    latmax='49.0'
    OUT_PATH='/nobackup/rossby26/users/sm_fuxwa/AI/'$DOMAIN'/original/'$GCM/

elif [[ "$DOMAIN" == "ALP" ]]; then

    #ALP domain
    lonmin='-180.0'
    lonmax='180.0'
    latmin='-90.0'
    latmax='90.0'
    OUT_PATH='/nobackup/rossby26/users/sm_fuxwa/AI/'$DOMAIN'/original/'$GCM/
fi

# DAYHHMM

if [[ "$freq_in" == "3hr" ]]; then
    # for ta500..950, hus500..950, ua500..950, va500..950, mrsol (3hr)
    if [[ " ${VAR_LIST[*]} " == " ta "* ]] || [[ " ${VAR_LIST[*]} " == " hus "* ]] || \
        [[ " ${VAR_LIST[*]} " == " ua "* ]] || [[ " ${VAR_LIST[*]} " == " va "* ]] || \
        [[ " ${VAR_LIST[*]} " == " mrso "* ]] || [[ " ${VAR_LIST[*]} " == " mrsol "* ]] ; then
        FIRST_DAYHHMM_IN=010000 # 3hr
        LAST_DAYHHMM_IN=312100  # 3hr
    else
        # for hfls  hfss  hus500  mrfso  mrso  mrsol  rlds  rlns  rsds  rsns  ta500  va850 (3hr)
        #FIRST_DAYHHMM_IN=010130 # 3hr
        #LAST_DAYHHMM_IN=312230  # 3hr
        FIRST_DAYHHMM_IN=010000 # 3hr
        LAST_DAYHHMM_IN=312100  # 3hr
    fi
elif [[ "$freq_in" == "6hr" ]]; then
    FIRST_DAYHHMM_IN=010000 # 6hr
    LAST_DAYHHMM_IN=311800  # 6hr
elif [[ "$freq_in" == "1hr" ]]; then
    if [[ " ${VAR_LIST[*]} " == *" pr "* ]] || [[ " ${VAR_LIST[*]} " == *" hfls "* ]]; then
        FIRST_DAYHHMM_IN=010030 # pr
        LAST_DAYHHMM_IN=312330  # pr
    else
        FIRST_DAYHHMM_IN=010000 # tas
        LAST_DAYHHMM_IN=312300  # tas
    fi
elif [[ "$freq_in" == "day" ]]; then
    FIRST_DAYHHMM_IN=01 # 6hr
    LAST_DAYHHMM_IN=31  # 6hr
elif [[ "$freq_in" == "orog" ]]; then
    FIRST_DAYHHMM_IN="" # fx
    LAST_DAYHHMM_IN=""  # rx
fi

if [[ "$freq_out" == "3hr" ]]; then
    if [[ " ${VAR_LIST[*]} " == *" pr "* ]]; then
        FIRST_DAYHHMM_OUT=010130 # 3hr
        LAST_DAYHHMM_OUT=312230  # 3hr
    else
        FIRST_DAYHHMM_OUT=010000 # 3hr
        LAST_DAYHHMM_OUT=312100  # 3hr
    fi
elif [[ "$freq_out" == "6hr" ]]; then
    if [[ " ${VAR_LIST[*]} " == *" pr "* ]]; then
        FIRST_DAYHHMM_OUT=010300 # 6hr
        LAST_DAYHHMM_OUT=312100  # 6hr
    else
        FIRST_DAYHHMM_OUT=010000 # 6hr
        LAST_DAYHHMM_OUT=311800  # 6hr
    fi
elif [[ "$freq_out" == "day" ]]; then
    FIRST_DAYHHMM_OUT=01 # 6hr
    LAST_DAYHHMM_OUT=31  # 6hr
elif [[ "$freq_out" == "fx" ]]; then
    FIRST_DAYHHMM_OUT="" # fx
    LAST_DAYHHMM_OUT=""  # rx
fi


for ivar in ${VAR_LIST[@]} ; do

  yy=${FIRST_YEAR}
  while [ ${yy} -le ${LAST_YEAR} ]; do

    mm=${FIRST_MONTH}
    while [ ${mm} -le ${LAST_MONTH} ]; do

      if [[ ${#mm} -lt 2 ]] ; then
	mm2d="0${mm}"
      else
	mm2d="${mm}"
      fi

      let mm=mm+1
      #while [[ ${#mm} -lt 2 ]] ; do
      #  mm="0${mm}"
      #done

    done #mm

    indir=${indir0}/${freq_in}/${ivar}
    outdir=${OUT_PATH}${experiment_cmorized}/${freq_out}/${ivar}
    if [ ! -e ${outdir} ] ; then
        mkdir -p ${outdir}
    fi
    infile=${ivar}_${experiment}_${freq_in}_${yy}${FIRST_MONTH}${FIRST_DAYHHMM_IN}'-'${yy}${LAST_MONTH}${LAST_DAYHHMM_IN}'.nc' 
    outfile_smalldomain=${ivar}_${experiment}_${freq_in}_${yy}${FIRST_MONTH}${FIRST_DAYHHMM_IN}'-'${yy}${LAST_MONTH}${LAST_DAYHHMM_IN}'_smalldomain.nc'
    outfile_newfreq=${ivar}_${experiment}_${freq_out}_${yy}${FIRST_MONTH}${FIRST_DAYHHMM_OUT}'-'${yy}${LAST_MONTH}${LAST_DAYHHMM_OUT}'_smalldomain.nc'
    outfile_newfreq_tmp=${ivar}_${experiment}_${freq_out}_${yy}${FIRST_MONTH}${FIRST_DAYHHMM_OUT}'-'${yy}${LAST_MONTH}${LAST_DAYHHMM_OUT}'_smalldomain_tmp'

    cdo sellonlatbox,$lonmin,$lonmax,$latmin,$latmax $indir/$infile $outdir/$outfile_smalldomain

    if [[ "$freq_in" == "1hr" ]] && [[ "$freq_out" == "3hr" ]] ; then
        if [[ " ${VAR_LIST[*]} " == *" pr "* ]]; then
            cdo timselmean,3 $outdir/$outfile_smalldomain $outdir/$outfile_newfreq
            echo 'processing pr'
        else
            echo 'processing' ${ivar}
            #for hour in {1..8808..3}; do
            #    echo hour, $hour
            #    cdo select,timestep=$hour $outdir/$outfile_smalldomain $outdir/${outfile_newfreq_tmp}_${hour}'.nc'
            #done
            #cdo mergetime $outdir/${outfile_newfreq_tmp}_* $outdir/$outfile_newfreq
            #rm -f $outdir/${outfile_smalldomain_tmp}_*

            #cdo select,timestep=$(seq 1 3 8808 | shuf | tr '\n' ',' | sed '$s/,$/\n/') tas_12km_1hr_200001010000-200912312300.nc tas_12km_3hr_200001010000-200912312300.nc
            cdo select,timestep=$(seq 1 3 8808 | tr '\n' ',' | sed '$s/,$/\n/') $outdir/$outfile_smalldomain $outdir/$outfile_newfreq
        fi
    elif [[ "$freq_in" == "1hr" ]] && [[ "$freq_out" == "6hr" ]] ; then
        if [[ " ${VAR_LIST[*]} " == *" pr "* ]]; then
            cdo timselmean,6 $outdir/$outfile_smalldomain $outdir/$outfile_newfreq
            echo 'processing pr'
        else
            echo 'processing' ${ivar}
            cdo select,timestep=$(seq 1 6 8808 | tr '\n' ',' | sed '$s/,$/\n/') $outdir/$outfile_smalldomain $outdir/$outfile_newfreq
        fi
    elif [[ "$freq_in" == "3hr" ]] && [[ "$freq_out" == "6hr" ]] ; then
        if [[ " ${VAR_LIST[*]} " == *" pr "* ]]; then
            cdo timselmean,2 $outdir/$outfile_smalldomain $outdir/$outfile_newfreq
            echo 'processing pr'
        else
            echo 'processing' ${ivar}
            cdo select,timestep=$(seq 1 2 3000 | tr '\n' ',' | sed '$s/,$/\n/') $outdir/$outfile_smalldomain $outdir/$outfile_newfreq
        fi
    elif [[ "$freq_in" == "3hr" ]] && [[ "$freq_out" == "day" ]] ; then
        if [[ " ${VAR_LIST[*]} " == *" pr "* || " ${VAR_LIST[*]} " == *" sn "* ]]; then
            cdo timselmean,8 $outdir/$outfile_smalldomain $outdir/$outfile_newfreq
            echo 'processing pr'
        else
            echo 'processing' ${ivar}
            cdo select,timestep=$(seq 1 8 3000 | tr '\n' ',' | sed '$s/,$/\n/') $outdir/$outfile_smalldomain $outdir/$outfile_newfreq
        fi
    fi

    echo The current year-month is: $yy $mm2d
    let yy=yy+1 
  done #yy

  # Merge nc files
  outfile_all=${ivar}_${experiment}_${freq_out}_*${FIRST_MONTH}${FIRST_DAYHHMM_OUT}'-'*${LAST_MONTH}${LAST_DAYHHMM_OUT}'_smalldomain.nc'
  outfile_merge=${ivar}_${experiment}_${freq_out}_${FIRST_YEAR}${FIRST_MONTH}${FIRST_DAYHHMM_OUT}'-'${LAST_YEAR}${LAST_MONTH}${LAST_DAYHHMM_OUT}'.nc'
  outfile_merge_cmorized=${ivar}_${experiment_cmorized}_${freq_out}_${FIRST_YEAR}${FIRST_MONTH}${FIRST_DAYHHMM_OUT}'-'${LAST_YEAR}${LAST_MONTH}${LAST_DAYHHMM_OUT}'.nc'
  outfile_month=${ivar}_${experiment_cmorized}_${freq_out}_${NAMEMONTH}_${FIRST_YEAR}${FIRST_MONTH}${FIRST_DAYHHMM_OUT}'-'${LAST_YEAR}${LAST_MONTH}${LAST_DAYHHMM_OUT}'.nc'
  if [ -f $outdir/$outfile_merge ]; then
    rm  $outdir/$outfile_merge
  fi
  cdo mergetime $outdir/$outfile_all $outdir/$outfile_merge 
  rm $outdir/$outfile_all 
  rm $outdir/${ivar}_${experiment}_${freq_in}_*${FIRST_MONTH}${FIRST_DAYHHMM_IN}'-'*${LAST_MONTH}${LAST_DAYHHMM_IN}'_smalldomain.nc'
  mv $outdir/$outfile_merge  $outdir/$outfile_merge_cmorized
  cdo selmon,${SELMONTH} $outdir/$outfile_merge_cmorized $outdir/$outfile_month

done #ivar

