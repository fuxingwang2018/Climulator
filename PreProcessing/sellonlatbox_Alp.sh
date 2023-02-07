
#!/bin/bash

EXP='FPS12'
FIRST_MONTH=01
LAST_MONTH=12

if [[ "$EXP" == "FPS12" ]]; then
    # https://en.wikipedia.org/wiki/Module:Location_map/data/Alps
    # for tas, pr
    #indir0='/nobackup/rossby25/proj/rossby/joint_exp/eucp/netcdf/HCLIM38-ALADIN/ALP-12/ECMWF-ERAINT/evaluation/'
    #experiment='ALP-12_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-HCLIM38-ALADIN_v1'

    # for ta500..950, hus500..950, ua500..950, va500..950, phi500..950
    indir0='/nobackup/rossby26/users/sm_fuxwa/AI/CORDEX_FPS_ALP12_ERAI_CMORise' 
    #same as /nobackup/rossby25/proj/rossby/joint_exp/eucp/netcdf/HCLIM38-ALADIN/ALP-12/ECMWF-ERAINT/evaluation/ but different variables
    experiment='ALP-12_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-SMHI-HCLIM38-ALADIN_v1'
    experiment_cmorized='12km'
    FIRST_YEAR=2000
    LAST_YEAR=2009 #2014 available but 2009 to consistent with FPS3, 2009 discarded because of spinup
    #lonmin='4.5'
    #lonmax='17'
    #latmin='42.75'
    #latmax='48.5'
    freq_in='fx' #3hr, 6hr, fx
    freq_out='fx' #fx
    VAR_LIST=('orog' ) # orog
    #VAR_LIST=('tas' ) # pr, tas
    #VAR_LIST=(\
    #	'hus500' 'hus700' 'hus850' 'hus950' )
    #VAR_LIST=('ta500' 'ta700' 'ta850' 'ta950') # \
    #	'hus500' 'hus700' 'hus850' 'hus950' \
    #	'ua500' 'ua700' 'ua850' 'ua950' \
    #	'va500' 'va700' 'va850' 'va950')
    #VAR_LIST=('phi500' 'phi700' 'phi850' 'phi950')

elif [[ "$EXP" == "FPS3" ]]; then 
    indir0='/nobackup/rossby25/proj/rossby/joint_exp/eucp/CORDEX-FPSCONV/output/ALP-3/HCLIMcom/ECMWF-ERAINT/evaluation/r1i1p1/HCLIMcom-HCLIM38-AROME/fpsconv-x2yn2-v1/'
    experiment='ALP-3_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-HCLIM38-AROME_fpsconv-x2yn2-v1' 
    experiment_cmorized='3km'
    freq_in='1hr'
    freq_out='6hr'
    VAR_LIST=('tas')  # pr
    FIRST_YEAR=2000 # 1999 available but we discard it for spinup
    LAST_YEAR=2009
    #lonmin='9'
    #lonmax='13'
    #latmin='45.5'
    #latmax='47.7'
fi

SELMONTH=7
NAMEMONTH='July'

lonmin='9'
lonmax='13'
latmin='45.5'
latmax='47.7'

OUT_PATH='/nobackup/rossby26/users/sm_fuxwa/AI/'
# DAYHHMM
if [[ "$freq_in" == "3hr" ]]; then
    FIRST_DAYHHMM_IN=010000 # 3hr
    LAST_DAYHHMM_IN=312100  # 3hr
elif [[ "$freq_in" == "6hr" ]]; then
    FIRST_DAYHHMM_IN=010000 # 6hr
    LAST_DAYHHMM_IN=311800  # 6hr
elif [[ "$freq_in" == "1hr" ]]; then
    if [[ " ${VAR_LIST[*]} " == *" tas "* ]]; then
        FIRST_DAYHHMM_IN=010000 # tas
        LAST_DAYHHMM_IN=312300  # tas
    elif [[ " ${VAR_LIST[*]} " == *" pr "* ]]; then
        FIRST_DAYHHMM_IN=010030 # pr
        LAST_DAYHHMM_IN=312330  # pr
    fi
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

