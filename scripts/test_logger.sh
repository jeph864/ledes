#!/bin/bash
#SBATCH -J XRlinearEv                      # the job name
#SBATCH --nodes=2             # node count
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --mem=1G
#SBATCH -t 1:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1              # use 1 thread per taks
#SBATCH -N 1
#SBATCH --partition=informatik-mind
#SBATCH --output=output/logger_%j_out.txt         # capture output
#SBATCH --error=output/logger_%j_err.txt          # and error streams

#module purge
module add nvidia/10.0
module load anaconda3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh


conda activate deles
echo "$(pip list)"
PROJECT="/scratch/abijuru/mlp/ledes"
cd $PROJECT

#by default, use amazon
data=${1-amazon-670k}
python ledes/batch.py 
echo "Finished Testing the logger"
conda deactivate