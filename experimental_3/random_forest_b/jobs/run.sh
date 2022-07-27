echo 'HELLO BOX'
codedir=/work/frauke/ai4eo-hyperview/hyperview/random_forest
datadir=/work/shared_data/2022-ai4eo_hyperview
conda init
source ~/.bashrc
conda activate ai4eo_hyper
echo "conda env activated"

echo $codedir
cd $codedir

PYTHONPATH=$PYTHONPATH:"$codedir"
export PYTHONPATH

python3 rf_train.py --in-data $datadir --submission-dir $codedir/submissions --n-trials 500 --n-estimators 500 1200 --max-depth 200 500 --max-depth-none --min-samples-leaf 1 5 10 --regressors XGB --col-ix 0 --folds 5 --save-eval --eval-dir $codedir/evaluation --save-model --model-dir $codedir/models --eta 0.001 0.1 0.3 0.5 --min-child_weight 1 5 10 50 70 --objectice asym 

