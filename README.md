# Label Disentanglement in Partition-based Extreme Multilabel Classification(Ledes)
This projects implements the paper [Label Disentanglement in Partition-based Extreme Multilabel Classification](https://arxiv.org/pdf/2106.12751.pdf). 

# Requirements and Installation
- Python 3.7.13
- pip (>19.3)
- other dependencies can be found in the requirements.txt
## platforms  
- Linux (RHRK Cluster)
# Getting started
## installation  
- create a conda enviroment(for e.g deles). 
- install the requirements:
```bash  
pip install -r requirements.txt

```
## run the pipeline
- run to create output, dataset and model directories: 
```bash
mkdir -p output && mkdir -p dataset && mkdir -p model
```
- download the dataset and put in the dataset folder. necessary code to dowload the dataset can be found in the download_datasets.sh. Please remember to set the dataset name
- run the preprocessor: TODO (the feature matrix should be np.float32, remember to cast the type before training)
- train, predict and evaluate the new model: by running the run.sbatch script. using slurm:
```bash
 sbatch run.sbatch
```
**Note:** the dataset, the conda environmnet,  the project path, model names,... shouldcan/should be adjusted before running

## models  : TODO

## datasets  : TODO

