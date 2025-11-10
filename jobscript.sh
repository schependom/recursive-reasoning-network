#!/bin/sh 


### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J train-bachelor-thesis
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request x GB of system-memory
#BSUB -R "rusage[mem=32GB]"
### -- set the email address --
#BSUB -u s251739@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --


# Load environment variables
source .env

# Fail early if REPO is not set
if [ -z "${REPO}" ]; then
	echo "ERROR: REPO environment variable is not set. Check .env"
	exit 1
fi


echo "making directory for run"
date=$(date +%Y%m%d_%H%M)
mkdir -p "${REPO}/RRN/checkpoints/${date}"


# Activate venv
echo "Activating virtual environment"
module load python3/3.9.19
module load cuda/11.7
source ${REPO}/.venv/bin/activate


# run training
echo "Starting training"
python3 RRN/train.py $date


echo "Job finished at :"
date