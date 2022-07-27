#!/usr/bin/env python3

import os
from glob import glob
import numpy as np
import pandas as pd
import random
import time
from datetime import datetime
import argparse
from tqdm.auto import tqdm

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KernelDensity
import xgboost as xgb

import joblib
import optuna
from optuna.samplers import TPESampler

import sys

import smogn

class BaselineRegressor():
    """
    Baseline regressor, which calculates the mean value of the target from the training
    data and returns it for each testing sample.
    """
    def __init__(self):
        self.mean = 0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.mean = np.mean(y_train, axis=0)
        self.classes_count = y_train.shape[1]
        return self

    def predict(self, X_test: np.ndarray):
        return np.full((len(X_test), self.classes_count), self.mean)


class SpectralCurveFiltering():
    """
    Create a histogram (a spectral curve) of a 3D cube, using the merge_function
    to aggregate all pixels within one band. The return array will have
    the shape of [CHANNELS_COUNT]
    """

    def __init__(self, merge_function = np.mean):
        self.merge_function = merge_function

    def __call__(self, sample: np.ndarray):
        return self.merge_function(sample, axis=(1, 2))


def load_data(directory: str, file_path, istrain, args):
    """Load each cube, reduce its dimensionality and append to array.

    Args:
        directory (str): Directory to either train or test set
    Returns:
        [type]: A list with spectral curve for each sample.
    """
    datalist = []
    masklist = []
    aug_datalist = []
    aug_masklist = []
    aug_labellist = []

    if istrain:
        labels = load_gt(file_path, args)

    all_files = np.array(
        sorted(
            glob(os.path.join(directory, "*.npz")),
            key=lambda x: int(os.path.basename(x).replace(".npz", "")),
        )
    )

    # in debug mode, only consider first 100 patches
    if args.debug:
        all_files = all_files[:100]

    # only 11x11 patches
    #all_files = all_files[:650]


    for file_name in all_files:
        with np.load(file_name) as npz:
            mask = npz['mask']
            data = npz['data']
            datalist.append(data)
            masklist.append(mask)

    if istrain:
        for i in range(args.augment_constant):
            for idx, file_name in enumerate(all_files):
                with np.load(file_name) as npz:
                    mask = npz['mask']
                    data = npz['data']
                    ma = np.max(data, keepdims=True)
                    sh = data.shape[1:]
                    max_edge = np.max(sh)
                    min_edge = np.min(sh)  # AUGMENT BY SHAPE
                    edge = min_edge  # np.random.randint(16, min_edge)
                    x = np.random.randint(sh[0] + 1 - edge)
                    y = np.random.randint(sh[1] + 1 - edge)
                    aug_data = data[:, x:(x + edge), y:(y + edge)] + np.random.uniform(-0.01, 0.01,
                                                                                       (150, edge, edge)) * ma
                    aug_mask = mask[:, x:(x + edge), y:(y + edge)] | np.random.randint(0, 1, (150, edge, edge))
                    aug_datalist.append(aug_data)
                    aug_masklist.append(aug_mask)
                    aug_labellist.append(labels[idx, :] + labels[idx, :] * np.random.uniform(-0.01, 0.01, 4))

    if istrain:
        return datalist, masklist, labels, aug_datalist, aug_masklist, np.array(aug_labellist)
    else:
        return datalist, masklist

def load_gt(file_path: str, args):
    """Load labels for train set from the ground truth file.
    Args:
        file_path (str): Path to the ground truth .csv file.
    Returns:
        [type]: 2D numpy array with soil properties levels
    """
    gt_file = pd.read_csv(file_path)

    # in debug mode, only consider first 100 patches
    if args.debug:
        gt_file = gt_file[:100]    
    
    # only 11x11 patches
    #gt_file = gt_file[:650]

    labels = gt_file[["P", "K", "Mg", "pH"]].values/np.array([325.0, 625.0, 400.0, 7.8]) # normalize ground-truth between 0-1
    
    return labels

def preprocess(data_list, mask_list):
    
    def _shape_pad(data):
        
        max_edge = np.max(image.shape[1:])
        shape = (max_edge, max_edge)
        padded = np.pad(data,((0, 0),
                             (0, (shape[0] - data.shape[1])),
                             (0, (shape[1] - data.shape[2]))),
                             'wrap')
        return padded

    def _random_pixel(data):
        '''draws (min_sample_size x min_sample_size) patches from each patch''' 
        
        min_edge = 11
        shape = (min_edge, min_edge)

        random_select = [np.random.choice(data[i].flatten(), min_edge*min_edge, replace=False).reshape(shape) for i in range(data.shape[0])]
        random_select = np.array(random_select)

        return random_select

    filtering = SpectralCurveFiltering()

    processed_data = []

    for idx, (data, mask) in enumerate(tqdm(zip(data_list, mask_list), total=len(data_list), 
                                        position=0, leave=True, desc="INFO: Preprocessing data ...")):

        data = data/2210 ## max-max=5419 mean-max=2210
        m = (1 - mask.astype(int))
        image = (data * m)
        #image = _random_pixel(image) 
        image = _shape_pad(image)
        s = np.linalg.svd(image, full_matrices=False, compute_uv=False)

        data = np.ma.MaskedArray(data, mask)
        arr = filtering(data)

        # first gradient
        dXdl = np.gradient(arr, axis=0)

        # second gradient
        d2Xdl2 = np.gradient(dXdl, axis=0)

        # fourier transform
        fft = np.fft.fft(arr)
        real = np.real(fft)
        imag = np.imag(fft)

        # final input matrix
        out = np.concatenate([arr,dXdl, d2Xdl2, s[:,0], s[:,1], s[:,2], s[:,3], s[:,4], real, imag], -1)

        processed_data.append(out)

    return np.array(processed_data)

def assym_loss(y_val, y_pred):
    factor = 50.
    residual = (y_val - y_pred).astype("float")
    grad = np.where(residual<0, -2*factor*residual, -2*residual)
    hess = np.where(residual<0, 2*factor, 2.0)
    return grad, hess

def cubic(y_val, y_pred):
    grad = 4*(y_val -y_pred)**3
    hess = 12*(y_val - y_pred)**2
    return grad, hess

def rmse_weighted(y, y_hat):
    weights =  np.repeat(1, y.shape[0])#weights_loss(y)
    grad = 2*weights*(y-y_hat)
    hess = 2*weights
    return grad, hess

def weights_loss(y, eps=0.0001, alpha=1.):
    '''
    function to calculate normalized weights based on the kernel estimation or each soil paramter separately
    These weights are the used for RMSE loss
    '''
    y = y.reshape(-1,1)
    kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(y)
    # score_samples returns the log of the probability density
    x_d = np.linspace(np.min(y), np.max(y), y.shape[0])
    logprob = kde.score_samples(x_d[:, None])
    distr = np.exp(logprob)
    distr_prime = (distr - np.min(distr, axis=0))/(np.max(distr, axis=0) - np.min(distr, axis=0))
    eps_array = np.zeros(distr_prime.shape)
    eps_array.fill(eps)
    weights = np.max([(1 - alpha*distr_prime), eps_array], axis=0)
    
    norm = 1/distr_prime.shape[0] * np.sum(np.max([(1 - alpha*distr_prime), eps_array], axis=0))

    weights_norm = weights/norm

    return weights_norm

def mixing_augmentation(X, y, fract, mix_const):

    mix_index_1 = np.random.randint(X.shape[0], size=int(np.floor(X.shape[0]*fract)))
    mix_index_2 = np.random.randint(X.shape[0], size=int(np.floor(X.shape[0]*fract)))

    ma_X = (1 - mix_const) * X[mix_index_1] + mix_const * (X[mix_index_2])
    ma_y = (1 - mix_const) * y[mix_index_1] + mix_const * (y[mix_index_2])

    return np.concatenate([X, ma_X], 0), np.concatenate([y, ma_y], 0)


def smogn_overs(X, y, col):
        ''' oversampling '''
        train = np.concatenate([X, y[:,col].reshape(-1,1)],  axis=1)
        features = np.arange(X.shape[1])
        df = pd.DataFrame(train, columns=(list(features) + ['y']))
        hyper_smogn = smogn.smoter(
            data = df,
            y = 'y'
            )
        return hyper_smogn
 


def evaluation_score(args, y_v, y_hat, y_b, cons):
    score = 0
    for i in range(len(args.col_ix)):
        print(f'Soil idx {i} / {len(args.col_ix)-1}')
        if len(args.col_ix) == 1:
            y_v = y_v.reshape(-1,1)
            y_hat = y_hat.reshape(-1,1)
        mse_model = mean_squared_error(y_v[:, i]*cons[i], y_hat[:, i]*cons[i])
        mse_bl = mean_squared_error(y_v[:, i]*cons[i], y_b[:, i]*cons[i])

        score += mse_model / mse_bl

        print(f'Baseline MSE:      {mse_bl:.2f}')
        print(f'Model MSE: {mse_model:.2f} ({1e2*(mse_model - mse_bl)/mse_bl:+.2f} %)')
        print(f'Evaluation score: {score/len(args.col_ix)}')
    
    return score / len(args.col_ix)       

def print_feature_importances(feature_names, importances):
    
    feats = {}
    for feature, importance in zip(feature_names, importances):
         feats[feature] = importance
    feats = sorted(feats.items(), key=lambda x: x[1], reverse=True)
    for feat in feats:
        print(f'{feat[0]}: {feat[1]}')



def predictions_and_submission(study, X_processed, X_test, y_train_col, cons, args):
   
    final_model = study.best_params["regressor"]
    if final_model == "RandomForest":
        # fit rf with best parameters on entire training data 
        optimised_model = RandomForestRegressor(n_estimators=study.best_params['n_estimators'], 
                                             max_depth=study.best_params['max_depth'],
                                             min_samples_leaf=study.best_params['min_samples_leaf'],
                                             n_jobs=-1, 
                                             criterion="squared_error")
    else:
        parameters = {"objective": 'reg:squarederror',
                      "n_estimators": study.best_params['n_estimators'],
                      "eta": study.best_params['eta'],
                      "gamma": study.best_params['gamma'],
                      "alpha": study.best_params['alpha'],
                      "max_depth": study.best_params['max_depth'],
                      "min_child_weight": study.best_params['min_child_weight'],
                      "verbosity": 1}

        if len(args.col_ix)==1:
            if args.objective == 'weighted_mse':
                parameters["objective"] = rmse_weighted
            if args.objective == 'cubic':
                parameters["objective"] = cubic
            if args.objective == 'asym':
                parameters["objective"] = assym_loss
            optimised_model = xgb.XGBRegressor(**parameters)

        else:
            optimised_model = MultiOutputRegressor(xgb.XGBRegressor(**parameters))

    if len(args.col_ix) == 1:
        y_train_col = y_train_col.ravel()

    optimised_model.fit(X_processed, y_train_col)
    predictions = optimised_model.predict(X_test)
   

    predictions = predictions * np.array(cons[:len(args.col_ix)])
    
    # calculate score on full training set
    baseline = BaselineRegressor()
    if len(args.col_ix) == 1:
        y_train_col = y_train_col.reshape(-1,1)
    baseline.fit(X_processed, y_train_col)
    y_b = baseline.predict(X_processed)
    y_fulltrain_pred = optimised_model.predict(X_processed)

    score = evaluation_score(args, y_train_col, y_fulltrain_pred, y_b, cons)
    print(f'\nScore of best model ({final_model}) on training set: {score}\n')

    # print feature importances for RF
    feature_names=['arr','dXdl', 'd2Xdl2', 's_0', 's_1', 's_2', 's_3', 's_4', 'real', 'imag']
    if final_model == 'RandomForest' or len(args.col_ix)==1:
        importances = optimised_model.feature_importances_
        print_feature_importances(feature_names, importances)
    elif final_model == 'XGB' and len(args.col_ix) > 1:
        for i in range(len(optimised_model.estimators_)):
            importances = optimised_model.estimators_[i].feature_importances_
            
            print_feature_importances(feature_names, importances)
    
    # save the model
    if args.save_model:
        if final_model=="RandomForest":
            output_file = os.path.join(args.model_dir, f"{final_model}_SIMPLE_ix={args.col_ix}_{date_time}_"\
                f"nest={study.best_params['n_estimators']}_maxd={study.best_params['max_depth']}_"\
                f"minsl={study.best_params['min_samples_leaf']}_"\
                f"_aug_con={study.best_params['augment_constant']}_"\
                f"aug_par={study.best_params['augment_partition']}"\
                f".bin")
        else:
            output_file = os.path.join(args.model_dir, f"{final_model}_SIMPLE_ix={args.col_ix}_"\
                f"{date_time}_nest={study.best_params['n_estimators']}_maxd={study.best_params['max_depth']}_"\
                f"eta={study.best_params['eta']}_"\
                f"gamma={study.best_params['gamma']}_"\
                f"alpha={study.best_params['alpha']}_"\
                f"minsl={study.best_params['min_child_weight']}_"
                f"aug_con={study.best_params['augment_constant']}_"\
                f"aug_par={study.best_params['augment_partition']}"\
                f".bin")


            
        with open(output_file, "wb") as f_out:
            joblib.dump(optimised_model, f_out)

    # only make submission file, if all 4 soil parameters are considered
    if len(args.col_ix) == 4 and  args.debug==False:
        submission = pd.DataFrame(data=predictions, columns=["P", "K", "Mg", "pH"])
        print(submission.head())
        if final_model=="RandomForest":
            submission.to_csv(os.path.join(args.submission_dir, f"submission_{final_model}_SIMPLE"\
                f"{date_time}_nest={study.best_params['n_estimators']}_maxd={study.best_params['max_depth']}_"\
                f"minsl={study.best_params['min_samples_leaf']}_"\
                f"aug_con={study.best_params['augment_constant']}_"\
                f"aug_par={study.best_params['augment_partition']}"\
                f".csv"), index_label="sample_index")
        else:
            submission.to_csv(os.path.join(args.submission_dir, f"submission_{final_model}_SIMPLE"\
                f"{date_time}_nest={study.best_params['n_estimators']}_maxd={study.best_params['max_depth']}_"\
                f"eta={study.best_params['eta']}_"\
                f"gamma={study.best_params['gamma']}_"\
                f"alpha={study.best_params['alpha']}_"\
                f"minsl={study.best_params['min_child_weight']}_"
                f"aug_con={study.best_params['augment_constant']}_"\
                f"aug_par={study.best_params['augment_partition']}"\
                f".csv"), index_label="sample_index")


def predictions_and_submission_2(study, best_model, X_test, cons, args, min_score):

    predictions = []
    for rf in best_model:
        pp = rf.predict(X_test)
        predictions.append(pp)
    predictions = np.asarray(predictions)
    predictions = np.mean(predictions, axis=0)
    
    predictions = predictions * np.array(cons[:len(args.col_ix)])


    final_model = best_model[0].__class__.__name__
    # print feature importances for Random Forest
    if final_model == "RandomForestRegressor":
        feats = {}
        importances = best_model[-1].feature_importances_
        feature_names = ['arr', 'dXdl', 'd2Xdl2', 'd3Xdl3', 'dXds1', 's_0', 
                         's_1', 's_2', 's_3', 's_4', 'real', 'imag']
        #feature_names = ['arr', 'dXdl', 'd2Xdl2', 'd3Xdl3', 'dXds1', 's_0', 
        #                 's_1', 's_2', 's_3', 's_4', 'real', 'imag',
        #                 'reals','imags', 'cDw2', 'cAw2', 'cos']
        for feature, importance in zip(feature_names, importances):
            feats[feature] = importance
        feats = sorted(feats.items(), key=lambda x: x[1], reverse=True)
        for feat in feats:
            print(f'{feat[0]}: {feat[1]}')

    # only make submission file, if all 4 soil parameters are considered
    if len(args.col_ix) == 4 and args.debug == False:
        submission = pd.DataFrame(data=predictions, columns=["P", "K", "Mg", "pH"])
        print(submission.head())
        if study is not None:
            if final_model=="RandomForestRegressor":
                submission.to_csv(os.path.join(args.submission_dir, f"submission_{final_model}_CV"\
                        f"{date_time}_nest={study.best_params['n_estimators']}_"\
                        f"maxd={study.best_params['max_depth']}_" \
                        f"minsl={study.best_params['min_samples_leaf']}"\
                        f"_aug_con={study.best_params['augment_constant']}_"\
                        f"aug_par={study.best_params['augment_partition']}"\
                        f".csv"),index_label="sample_index")
            else:
                submission.to_csv(os.path.join(args.submission_dir, f"submission_{final_model}_CV"\
                        f"{date_time}_nest={study.best_params['n_estimators']}_"\
                        f"maxd={study.best_params['max_depth']}_" \
                        f"eta={study.best_params['eta']}_"\
                        f"gamma={study.best_params['gamma']}_"\
                        f"alpha={study.best_params['alpha']}_"\
                        f"aug_con={study.best_params['augment_constant']}_"\
                        f"aug_par={study.best_params['augment_partition']}"\
                        f".csv"),index_label="sample_index")

        else:
            submission.to_csv(os.path.join(args.submission_dir, "submission_best_{}.csv".format(min_score)),
                    index_label="sample_index")


def main(args):

    train_data = os.path.join(args.in_data, "train_data", "train_data")
    test_data = os.path.join(args.in_data, "test_data")
    train_gt=os.path.join(args.in_data, "train_data", "train_gt.csv")

    # load the data
    print("start loading data ...")
    start_time = time.time()
    X_train, M_train, y_train, X_aug_train, M_aug_train, y_aug_train = load_data(train_data, train_gt, True, args)
    print(f"loading train data took {time.time() - start_time:.2f}s")
    print(f"train data size: {len(X_train)}")
    if args.debug==False:
        print(f"patch size examples: {X_train[0].shape}, {X_train[500].shape}, {X_train[1000].shape}")
    
    start_time = time.time()
    X_test, M_test = load_data(test_data, None, False, args)
    print(f"loading test data took {time.time() - start_time:.2f}s")
    print(f"test data size: {len(X_test)}\n")
    
    print('Preprocess training data...')
    X_processed = preprocess(X_train, M_train)
    X_aug_processed = preprocess(X_aug_train, M_aug_train)

    print('preprocess test data ...')
    X_test = preprocess(X_test, M_test)
    
    # selected set of labels
    y_aug_train_col = y_aug_train[:, args.col_ix]
    y_train_col = y_train[:, args.col_ix]
    y_train_col_copy = y_train_col

    cons = np.array([325.0, 625.0, 400.0, 7.8])
 
    # oversampling - only for simple regression!!
    if args.smogn and len(args.col_ix)==1:
        smogn_train = smogn_overs(X_processed, y_train_col, args.col_ix[0])
        print(f'smogn on idx {args.col_ix[0]}')
        print(f'train data shape after smogn: {smogn_train.shape}')
        smogn_train = np.array(smogn_train)
        X_processed = smogn_train[:,:-1]
        y_train_col = smogn_train[:,-1].reshape(-1,1)
        print(f'X train data shape after smogn: {X_processed.shape}')
        print(f'y train data shape after smogn: {y_train_col.shape}')

    global best_model
    best_model = None
    global min_score
    min_score = np.inf
    global y_hat_bls 
    y_hat_bls = []
    global y_hat_rfs
    y_hat_rfs = []
    global y_vs 
    y_vs = []


    def objective(trial):
        global best_model
        global min_score

        print(f"\nTRIAL NUMBER: {trial.number}\n")
        # training
        kfold = KFold(n_splits=args.folds, shuffle=True, random_state=RANDOM_STATE)
    
        random_forests = []
        baseline_regressors = []
        scores = []

        print("START TRAINING ...")
        for i, (ix_train, ix_valid) in enumerate(kfold.split(np.arange(0, len(y_train_col)))):
        
            print(f'fold {i}:')

            X_t = X_processed[ix_train]
            y_t = y_train_col[ix_train]
     
            augment_constant = trial.suggest_int('augment_constant', 0, args.augment_constant, log=False)
            augment_partition = trial.suggest_int('augment_partition', args.augment_partition[0], args.augment_partition[1], log=True)

            for idy in range(augment_constant):
                X_ta_1 = X_aug_processed[ix_train+(idy*len(y_train))]
                y_ta_1 = y_aug_train_col[ix_train+(idy*len(y_train))]
                X_t=np.concatenate((X_t,X_ta_1[-augment_partition:]),axis=0)
                y_t=np.concatenate((y_t,y_ta_1[-augment_partition:]),axis=0)


            # mixing augmentation
            if args.mix_aug:
                fract = trial.suggest_categorical('fract', args.fract)
                mix_const = trial.suggest_float('mix_const', args.fract)
                X_t, y_t = mixing_augmentation(X_t, y_t, fract, mix_const)

            X_v = X_processed[ix_valid]
            y_v = y_train_col[ix_valid]

            # baseline
            baseline = BaselineRegressor()
            baseline.fit(X_t, y_t)
            baseline_regressors.append(baseline)

            reg_name= trial.suggest_categorical("regressor", args.regressors)

            print(f"Training on {reg_name}")
            if reg_name == "RandomForest":
                n_estimators =  trial.suggest_int('n_estimators', args.n_estimators[0], args.n_estimators[1], log=True)
                max_depth =  trial.suggest_categorical('max_depth', args.max_depth)
                min_samples_leaf =  trial.suggest_categorical('min_samples_leaf', args.min_samples_leaf)

                # random forest
                model = RandomForestRegressor(n_estimators=n_estimators, 
                                           max_depth=max_depth, 
                                           min_samples_leaf=min_samples_leaf, 
                                           n_jobs=-1, 
                                           criterion="squared_error")
            else:
                n_estimators =  trial.suggest_int('n_estimators', args.n_estimators[0], args.n_estimators[1], log=True)
                eta = trial.suggest_float('eta', args.eta[0], args.eta[1], log=True)
                gamma = trial.suggest_float('gamma', args.gamma[0], args.gamma[1])
                alpha = trial.suggest_float('alpha', args.alpha[0], args.alpha[1])
                max_depth = trial. suggest_categorical('max_depth', args.max_depth)
                min_child_weight = trial. suggest_categorical('min_child_weight', args.min_child_weight)


                # xgboost
                parameters = {"objective": 'reg:squarederror',
                              "n_estimators": n_estimators,
                              "eta": eta,
                              "gamma": gamma,
                              "alpha": alpha,
                              "max_depth": max_depth,
                              "min_child_weight": min_child_weight,
                              "verbosity": 1}

                if len(args.col_ix)==1:
                    y_t = y_t.ravel()
                    if args.objective == 'weighted_mse':
                        parameters["objective"] = rmse_weighted
                    if args.objective == 'cubic':
                        parameters["objective"] = cubic
                    if args.objective == 'asym':
                        parameters["objective"] = assym_loss
                    
                    model = xgb.XGBRegressor(**parameters)

                else:
                    model = MultiOutputRegressor(xgb.XGBRegressor(**parameters))
            print(f"loss function used: {parameters['objective']}")
            model.fit(X_t, y_t)
            random_forests.append(model)
            print(f'{reg_name} score: {model.score(X_v, y_v)}')

            # predictions
            y_hat = model.predict(X_v)
            y_b = baseline.predict(X_v)

            # save y_hat and y_v for evaluation
            y_hat_bls.append(y_b)
            y_hat_rfs.append(y_hat)
            y_vs.append(y_v)


            # evaluation score
            score = evaluation_score(args, y_v, y_hat, y_b, cons)
            scores.append(score)
            print(scores)
    
        print("END TRAINING")
        # final score
        mean_score = np.mean(np.array(scores))
        print(f'mean score: {mean_score}\n')
        if mean_score < min_score:
            min_score = mean_score
            best_model = random_forests
            #predictions_and_submission_2(None, best_model, X_test, cons, args, min_score)

        return mean_score
    
    study = optuna.create_study(sampler=TPESampler(), direction='minimize')
    study.optimize(objective, n_trials=args.n_trials)

    # save study
    final_model = study.best_params["regressor"]

    if args.debug == False and final_model=="RandomForest":
        output_file = os.path.join(args.submission_dir, f"study_{final_model}_{date_time}_"\
                                        f"nest={study.best_params['n_estimators']}_"\
                                        f"maxd={study.best_params['max_depth']}_"\
                                        f"minsl={study.best_params['min_samples_leaf']}.pkl")
    if args.debug == False and final_model=="XGB":
        output_file = os.path.join(args.submission_dir, f"study_{final_model}_{date_time}_"\
                                        f"nest={study.best_params['n_estimators']}_"\
                                        f"maxd={study.best_params['max_depth']}_"\
                                        f"eta={study.best_params['eta']}_"\
                                        f"gamma={study.best_params['gamma']}_"\
                                        f"alpha={study.best_params['alpha']}_"\
                                        f"minsl={study.best_params['min_child_weight']}.pkl")

    if args.debug == False:
        with open(output_file, "wb") as f_out:
            joblib.dump(study, f_out)

    # prepare submission
    print("MAKE PREDICTIONS AND PREPARE SUBMISSION")
    # train best model on full training set
    predictions_and_submission(study, X_processed, X_test, y_train_col, cons, args)
    # cross validation on validation set
    predictions_and_submission_2(study, best_model, X_test, cons, args, min_score)
    print("PREDICTIONS AND SUBMISSION FINISHED")

    # save y_vs and y_hats
    y_hat_bls = np.concatenate(y_hat_bls, axis=0)
    y_hat_rfs = np.concatenate(y_hat_rfs, axis=0)
    y_vs = np.concatenate(y_vs, axis=0)
    if final_model == 'RandomForest':
        main_name = f"_{date_time}_"\
                    f"nest={study.best_params['n_estimators']}_"\
                    f"maxd={study.best_params['max_depth']}"\
                    f"minsl={study.best_params['min_samples_leaf']}"
        eval_name_bls = f"y_hat_bls_{final_model}_{date_time}_" + main_name
        eval_name_rfs = f"y_hat_rfs_{final_model}_{date_time}_" + main_name
        eval_name_vs = f"y_vs_{final_model}_{date_time}_" + main_name
    else:
         main_name = f"nest={study.best_params['n_estimators']}_"\
                    f"maxd={study.best_params['max_depth']}"\
                    f"eta={study.best_params['eta']}"\
                    f"gamma={study.best_params['gamma']}"\
                    f"alpha={study.best_params['alpha']}"\
                    f"mincw={study.best_params['min_child_weight']}"
         eval_name_bls = f"y_hat_bls_{final_model}_{date_time}_" + main_name
         eval_name_rfs = f"y_hat_rfs_{final_model}_{date_time}_" + main_name
         eval_name_vs = f"y_vs_{final_model}_{date_time}_" + main_name

    if args.save_eval:
        np.save(os.path.join(args.eval_dir, eval_name_bls), y_hat_bls)
        np.save(os.path.join(args.eval_dir, eval_name_rfs), y_hat_rfs)
        np.save(os.path.join(args.eval_dir, eval_name_vs), y_vs)

if __name__ == "__main__":

    RANDOM_STATE = 42
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    
    now = datetime.now()
    date_time = now.strftime("%Y%m%d%H%M")

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--in-data', type=str, 
            default='/mnt/lustre02/work/ka1176/shared_data/2022-ai4eo_hyperview')
    parser.add_argument('--submission-dir', type=str, 
            default='/mnt/lustre02/work/ka1176/frauke/ai4eo-hyperview/hyperview/random_forest/submissions')
    parser.add_argument('--model-dir', type=str, 
            default='/mnt/lustre02/work/ka1176/frauke/ai4eo-hyperview/hyperview/random_forest/models')
    parser.add_argument('--eval-dir', type=str, 
            default='/mnt/lustre02/work/ka1176/frauke/ai4eo-hyperview/hyperview/random_forest/evaluation')
    parser.add_argument('--save-model', action='store_true', default=False)
    parser.add_argument('--save-pred', action='store_true', default=False)
    parser.add_argument('--save-eval', action='store_true', default=False)
    parser.add_argument('--col-ix', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--folds', type=int, default=5)
    # model hyperparams
    parser.add_argument('--n-estimators', type=int, nargs='+', default=[500, 1000])
    parser.add_argument('--max-depth', type=int, nargs='+', default=[5, 10, 100, None])
    parser.add_argument('--max-depth-none', action='store_true', default=False)
    parser.add_argument('--min-samples-leaf', type=int, nargs='+', default=[1, 10, 50])
    parser.add_argument('--eta', type=float, nargs='+', default=[0.1, 0.5]) # default 0.3
    parser.add_argument('--gamma', type=float, nargs='+', default=[0, 1]) # default=0
    parser.add_argument('--alpha', type=float, nargs='+', default=[0, 1]) # default=0
    parser.add_argument('--min-child_weight', type=int, nargs='+', default=[1, 10, 50])
    parser.add_argument('--regressors', type=str, nargs='+', default=["RandomForest"])
    parser.add_argument('--n-trials', type=int, default=100)
    # augmentation
    parser.add_argument('--mix-aug', action='store_true', default=False)
    parser.add_argument('--fract', type=float, nargs='+', default=[0.1])
    parser.add_argument('--mix-const', type=float, nargs='+', default=[0.05])
    parser.add_argument('--augment-constant', type=int, default=5)
    parser.add_argument('--augment-partition', type=int, nargs='+', default=[100, 350])
    parser.add_argument('--smogn', action='store_true', default=False)
    parser.add_argument('--weights', action='store_true', default=False)
    parser.add_argument('--objective', type=str, default='mse', choices=['mse', 'weighted_mse', 'cubic', 'asym'])


    args = parser.parse_args()

    # None is added to max-depth (annot be done directly -> type error)
    if args.max_depth_none:
        args.max_depth = args.max_depth + [None]

    print('BEGIN argparse key - value pairs')
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    print('END argparse key - value pairs')
    print()

    if args.smogn:
        assert len(args.col_ix) == 1, 'smogn oversampling only for simple output regression (len(col_ix==1)) possible'

    cols = ["P205", "K", "Mg", "pH"]
    
    feature_names = ['arr', 'dXdl', 'd2Xdl2', 'd3Xdl3', 'dXds1', 's_0',
                         's_1', 's_2', 's_3', 's_4', 'real', 'imag']

    main(args)

