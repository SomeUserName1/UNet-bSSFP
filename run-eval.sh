#!/bin/bash -l
 
##############################
#       Job blueprint        #
##############################
 
# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=eval-models
 
# Define, how many nodes you need. Here, we ask for 1 node.
# Each node has 16 or 20 CPU cores.
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --cpus-per-task=128

# You can further define the number of tasks with --ntasks-per-*
# See "man sbatch" for details. e.g. --ntasks=4 will ask for 4 cpus.
 
# Define, how long the job will run in real time. This is a hard cap meaning
# that if the job runs longer than what is written here, it will be
# force-stopped by the server. If you make the expected time too long, it will
# take longer for the job to start. Here, we say the job will take 5 minutes.
#              d-hh:mm:ss
#SBATCH --time=3-00:05:00
 
# Define the partition on which the job shall run. May be omitted.
#SBATCH --partition=highmem
# #SBATCH --gres=gpu:
 
# How much memory you need.
# --mem will define memory per node and
# --mem-per-cpu will define memory per CPU/core. Choose one of those.
# #SBATCH --mem-per-cpu=512MB
##SBATCH --mem=128G    # this one is not in effect, due to the double hash
#SBATCH --exclusive
 
# Turn on mail notification. There are many possible self-explaining values:
# NONE, BEGIN, END, FAIL, ALL (including all aforementioned)
# For more values, check "man sbatch"
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fabian.klopfer@ieee.org
 
# You may not place any commands before the last SBATCH directive

# You can copy everything you need to the scratch directory
# ${SLURM_SUBMIT_DIR} points to the path where this script was submitted from
 
# This is where the actual work is done. In this case, the script only waits.
# The time command is optional, but it may give you a hint on how long the
# command worked
export OMP_NUM_THREADS=1
srun singularity exec --bind /ptmp/fklopfer/:/ptmp/fklopfer/ --bind /home/fklopfer/:/home/fklopfer/ /ptmp/containers/python38.sif bash -c "python3 /home/fklopfer/UNet-bSSFP/src/eval.py"
 
# Finish the script
exit 0
