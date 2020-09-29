#!/bin/bash
#SBATCH --job-name=pca_analysis.py
#SBATCH --partition=q36,q28,q24,q20
#SBATCH --mem=250G
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=out/slurm-%A_%a.out
#SBATCH --no-requeue

echo "========= Job started  at `date` =========="

source /home/abb/Python3.6/bin/activate
source /home/abb/.initialization
export DFTB_PREFIX="/home/sm/DFT/DFTB/dftb_parameters/organic/"
export DFTB_COMMAND="/home/lassebv/python/bin/dftb+_1.2.1.x86_64-linux"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
NUM_CORES="$(nproc)"

echo "My jobid: $SLURM_JOB_ID"
echo "I have 1 core(s)!"

MAINDIR="$(pwd)"
JOBDIR="/scratch/$SLURM_JOB_ID/"
cd $JOBDIR

python $MAINDIR/pca_analysis.py 

cd $MAINDIR/
cp -r $JOBDIR/* . > /dev/null 2>&1

echo "========= Job finished at `date` =========="

