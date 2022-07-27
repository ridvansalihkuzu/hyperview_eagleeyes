#! /usr/bin/env python3

import os
import joblib
import argparse
import time
import numpy as np
import pandas as pd
from rf_train import load_data, preprocess

def make_submission(preds, model_names, args):
    
    names = []
    for model in model_names:
        tmp = model.split('.')[0]
        model_name = tmp.split('_')[4:]
        names.append('_'.join(model_name))

    submission = pd.DataFrame.from_dict(data=preds)
    submission.rename(columns = {'ix_0':'P', 'ix_1':'K', 'ix_2':'Mg', 'ix_3':'pH'}, inplace = True)
    print(submission.head())
    submission.to_csv(os.path.join(args.submission_dir, f"submission_HYBRID_XGB"\
            #f"model_ix0={names[0]}_"\
            #    f"model_ix1={names[1]}_"\
            #    f"model_ix2={names[2]}_"\
            #    f"model_ix3={names[3]}"\
                f".csv"), index_label="sample_index")


def main(args):

    # read and data
    test_data = os.path.join(args.indata, "test_data")

    start_time = time.time()
    X_test, M_test = load_data(test_data, None, False, args)
    print(f"loading test data took {time.time() - start_time:.2f}s")
    print(f"test data size: {len(X_test)}\n")

    print('preprocess test data ...')
    X_test = preprocess(X_test, M_test)

    cons = np.array([325.0, 625.0, 400.0, 7.8])

    # load models
    preds = {}
    model_names = []
    for i in range(4):
        model_ix = os.path.join(args.model_path, args.models[i])
        with open(model_ix, 'rb') as f_in:
            model = joblib.load(f_in)

            # make predictions of i-th column
            predictions = model.predict(X_test)
            preds[f'ix_{i}'] = predictions * np.array(cons[i])
            model_names.append(model_ix)

    print(preds.keys)
    print(preds)
    print(model_names)
    # make submission
    make_submission(preds, model_names, args)
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default="models")
    parser.add_argument('--models', type=str, nargs='+', 
            default=['RandomForest_SIMPLE_ix=[0]_202205031318_nest=901_maxd=None_minsl=10__aug_con=3_aug_par=271.bin',
                     'RandomForest_SIMPLE_ix=[1]_202204250701_nest=726_maxd=None_minsl=5__aug_con=5_aug_par=212.bin',
                     'RandomForest_SIMPLE_ix=[2]_202204250734_nest=729_maxd=None_minsl=5__aug_con=1_aug_par=343.bin',
                     'RandomForest_SIMPLE_ix=[3]_202205022205_nest=685_maxd=None_minsl=10__aug_con=5_aug_par=219.bin'],
            help='models in index order 0 to 3')

    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--indata', type=str, default='/mnt/lustre02/work/ka1176/shared_data/2022-ai4eo_hyperview/',
            help='data to make submission')
    parser.add_argument('--submission-dir', type=str,
            default='/mnt/lustre02/work/ka1176/frauke/ai4eo-hyperview/hyperview/random_forest/submissions')

    args = parser.parse_args()
    print('BEGIN argparse key - value pairs')
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    print('END argparse key - value pairs')
    print()

    main(args)
