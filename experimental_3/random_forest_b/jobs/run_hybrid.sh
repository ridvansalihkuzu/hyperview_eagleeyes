echo 'HELLO BOX'
codedir=/mnt/lustre02/work/ka1176/frauke/ai4eo-hyperview/hyperview/random_forest
datadir=/mnt/lustre02/work/ka1176/shared_data/2022-ai4eo_hyperview
conda init
source ~/.bashrc
conda activate ai4eo_hyper
echo "conda env activated"

echo $codedir
cd $codedir

PYTHONPATH=$PYTHONPATH:"$codedir"
export PYTHONPATH

python3 $codedir/rf_hybrid_submission.py --models 'XGB_SIMPLE_ix=[0]_202205172256_nest=977_maxd=None_eta=0.004479022461645764_gamma=0.08433258895627904_alpha=0.006492449700871317_minsl=50_aug_con=4_aug_par=263.bin' 'XGB_SIMPLE_ix=[1]_202205171930_nest=1117_maxd=None_eta=0.03118616043400089_gamma=0.0040393442019673945_alpha=0.026391135212343833_minsl=1_aug_con=5_aug_par=142.bin' 'XGB_SIMPLE_ix=[2]_202205171645_nest=948_maxd=None_eta=0.03551762239725967_gamma=0.001736102301409208_alpha=0.6389230993676236_minsl=10_aug_con=2_aug_par=126.bin' 'XGB_SIMPLE_ix=[3]_202205171543_nest=1106_maxd=None_eta=0.02499089040438502_gamma=0.0007804680973742435_alpha=0.19344057317568825_minsl=50_aug_con=5_aug_par=298.bin'
