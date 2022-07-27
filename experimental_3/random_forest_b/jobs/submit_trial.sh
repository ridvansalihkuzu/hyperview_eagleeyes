#!/bin/bash
#SBATCH --job-name=rf
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL
#SBATCH --account=k20200
#SBATCH --output=rf_out.o%j
#SBATCH --error=rf_err.e%j

conda init
source ~/.bashrc
conda activate ai4eo_hyper
echo "conda env activated"

# Run script
codedir=/mnt/lustre02/work/ka1176/frauke/ai4eo-hyperview/hyperview
datadir=/mnt/lustre02/work/ka1176/shared_data/2022-ai4eo_hyperview

PYTHONPATH=$PYTHONPATH:"$codedir"
export PYTHONPATH

python3 $codedir/random_forest/rf_train.py --in-data $datadir --submission-dir $codedir/random_forest/submissions 
