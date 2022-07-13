#!/bin/bash
#SBATCH -J xrlinearledes                      # the job name
#SBATCH --nodes=2             # node count
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --mem=40G
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16             # use 1 thread per taks
#SBATCH -N 1
#SBATCH --partition=informatik-mind
#SBATCH --output=output/1-fullpipeline_%j_out.txt         # capture output
#SBATCH --error=output/1-fullpipeline_%j_err.txt          # and error streams

#module purge
module add nvidia/10.0
module load anaconda3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh


conda activate deles
echo "$(pip list)"
PROJECT="/scratch/$USER/mlp/ledes"
cd $PROJECT


data=${1-amazon-670k}
i_model=${data}-1 #Model name after XMC-Part initialization
n_model=${data}-1-new # Model name after running labels disentanglement

echo "Training XR-Linear model "
echo "The dataset used is from ./dataset/xmc-base/${data}/"
python -m pecos.xmc.xlinear.train \
  -x ./dataset/xmc-base/${data}/tfidf-attnxml/X32.trn.npz \
  -y ./dataset/xmc-base/${data}/Y.trn.npz \
  -m ./model/${i_model} \
  --nr-splits 32 \
  -b 10


echo "Reorganizing clusters and renewing the matcher"
python ledes/model.py \
  -x ./dataset/xmc-base/${data}/tfidf-attnxml/X32.trn.npz \
  -y ./dataset/xmc-base/${data}/Y.trn.npz \
  -m model/${i_model}\
  -b 10 \
  --n_copies 2 \
  -o model/${n_model}

echo "Retrain our  our model"
python -m pecos.xmc.xlinear.train \
  -x ./dataset/xmc-base/${data}/tfidf-attnxml/X32.trn.npz \
  -y ./model/${n_model}/Y.trn.npz \
  -c ./model/${n_model} \
  -m ./model/${n_model} \
  -b 10
echo "Evaluating the model"
python ledes/evaluate.py \
  -x dataset/xmc-base/${data}/tfidf-attnxml/X.tst.npz \
  -m model/${n_model}  \
  -k 10 \
  -y dataset/xmc-base/${data}/Y.tst.npz  \
  -b 10 \
  --mapper model/${n_model}/pseudo_label_mapping.pkl  \
  --unused-labels model/${n_model}/unused_labels.pkl
echo "Finished evaluating  the model"
conda deactivate