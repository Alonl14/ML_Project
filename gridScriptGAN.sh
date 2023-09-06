#! /bin/bash
#PBS -m n
#PBS -l walltime=12:00:00

#### script that run the python script, the MadGraph generator
echo "Installing python3>>>>>>"
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_101_ATLAS_20 x86_64-centos7-gcc11-opt"

echo "running the python script >>>>>>"
n_epoch=${parname1}
Directory=${parname2}

#### go to the directory where the files live
cd ${Directory}
echo "I am now in "${PWD}

time python LUXE_WGAN.py ${n_epoch}
