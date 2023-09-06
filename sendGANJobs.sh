#! /bin/bash

### how many jobs you want to submit, if -1, then submits jobs you set
nepoch=${1}
runid=0
### a counter value
b=1
### runId increased by one
runid=$(( $runid + $b ))
echo "runid: "$runid
### the place where the output and error file of the grid will live
DESTINATION="/storage/agrp/alonle/ClusterLogs"
    
### if main directory/run_id exists, delete
#if [[ -d "${DESTINATION}/run_$runid" ]]; then
#    echo "Found a directory with output ${DESTINATION}/run_$runid! Deleting the previous one."
#    rm -rf ${DESTINATION}/run_$runid
#fi
   
mkdir -p ${DESTINATION}/run_${runid}
#### from where you are submitting jobs
PRESENTDIRECTORY=${PWD}
#### submit jobs to the PBS system
qsub -l ngpus=1,mem=4gb -v parname1=${nepoch},parname2=${PRESENTDIRECTORY} -q N -N "run_"$runid -o "${DESTINATION}/run_"${runid} -e "${DESTINATION}/run_"${runid} gridScriptGAN.sh
### sleep for 1 s, so that there is no problem in submitting jobs to the grid
sleep 1s
    

