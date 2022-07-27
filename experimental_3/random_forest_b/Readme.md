# Train Random Forest

## Using conda interactive
* create environment from ai4eo_hyper.yml: ```conda env create -f ai4eo_hyper.yml```
* activate environmnt ```conda activate ai4eo_hyper```
* run script ```python3 r_train.py --in-data /mnt/lustre02/work/ka1176/shared_data/2022-ai4eo_hyperview --submission-dir /mnt/lustre02/work/ka1176/frauke/ai4eo-hyperview/hyperview/submissions```

## Start a batch job
* ```cd ai4eo-hyperview/hyperview/random_forest/jobs```
* ```sbatch submit_trial.sh```

## Use Singularity / Docker
* pull the docker image ```singularity pull docker://froukje/ai4eo-hyperview:rf```
* ```cd ai4eo-hyperview/hyperview/random_forest/jobs```
* ```sbatch submit_singularity_trial.sh```
