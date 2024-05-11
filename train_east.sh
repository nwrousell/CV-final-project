#!/bin/bash
#SBATCH --nodes=1               # node count
#SBATCH -p gpu --gres=gpu:1     # number of gpus per node
#SBATCH --ntasks-per-node=1     # total number of tasks across all nodes
#SBATCH --cpus-per-task=1       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=40G               # total memory (4 GB per cpu-core is default)
#SBATCH -t 24:00:00             # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin       # send email when job begins
#SBATCH --mail-type=end         # send email when job ends
#SBATCH --mail-user=noah_rousell@brown.edu

export PYTHONUNBUFFERED=TRUE
source env/bin/activate

cd east

python train.py --name $SLURM_JOB_NAME