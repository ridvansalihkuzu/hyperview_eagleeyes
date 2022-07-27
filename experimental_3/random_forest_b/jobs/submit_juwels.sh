#!/bin/bash
#SBATCH --job-name=rf
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL
#SBATCH --account=hai_cons_ee
#SBATCH --output=rf_out.o%j
#SBATCH --error=rf_err.e%j

conda init
source ~/.bashrc
conda activate ai4eo_hyper
echo "conda env activated"

# Run script
codedir=/p/project/hai_cons_ee/caroline/gitlab/ai4eo-hyperview/hyperview
datadir=/p/project/hai_cons_ee/data/ai4eo-hyperspectral/raw_data/

PYTHONPATH=$PYTHONPATH:"$codedir"
export PYTHONPATH

python $codedir/random_forest/rf_train.py --in-data $datadir --submission-dir $codedir/random_forest/submissions
