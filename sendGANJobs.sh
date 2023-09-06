#! /bin/bash

### how many jobs you want to submit, if -1, then submits jobs you set
nepoch=${1}
<<<<<<< HEAD
runid=0
=======

>>>>>>> f59de1b82d7b145772789b8fd547eda2823344a7
### a counter value
b=1
### runId increased by one
runid=$(( $runid + $b ))
echo "runid: "$runid
### the place where the output and error file of the grid will live
DESTINATION="/storage/agrp/alonle/ClusterLogs"
<<<<<<< HEAD
=======
### create the main directory if it does not exists
mkdir -p ${DESTINATION}/run_${runid}
>>>>>>> f59de1b82d7b145772789b8fd547eda2823344a7
    
### if main directory/run_id exists, delete
#if [[ -d "${DESTINATION}/run_$runid" ]]; then
#    echo "Found a directory with output ${DESTINATION}/run_$runid! Deleting the previous one."
#    rm -rf ${DESTINATION}/run_$runid
#fi
   
<<<<<<< HEAD
mkdir -p ${DESTINATION}/run_${runid}
#### from where you are submitting jobs
PRESENTDIRECTORY=${PWD}
#### submit jobs to the PBS system
qsub -l ngpus=1,mem=4gb -v parname1=${nepoch},parname2=${PRESENTDIRECTORY} -q N -N "run_"$runid -o "${DESTINATION}/run_"${runid} -e "${DESTINATION}/run_"${runid} gridScriptGAN.sh
=======
#### from where you are submitting jobs
PRESENTDIRECTORY=${PWD}
#### submit jobs to the PBS system
qsub -l ngpus=1,mem=10gb -v parname1=${nepoch},parname2=${PRESENTDIRECTORY} -q N -N "run_"$runid -o "${DESTINATION}/run_"${runid} -e "${DESTINATION}/run_"${runid} gridScriptGAN.sh
>>>>>>> f59de1b82d7b145772789b8fd547eda2823344a7
### sleep for 1 s, so that there is no problem in submitting jobs to the grid
sleep 1s
    

