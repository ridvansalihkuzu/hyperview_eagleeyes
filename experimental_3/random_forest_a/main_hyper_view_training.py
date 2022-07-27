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
import xgboost as xgb
import pywt
import joblib
import optuna
from optuna.samplers import TPESampler
from sklearn import preprocessing
from scipy.fftpack import dct
from sklearn.preprocessing import PowerTransformer
import sys
from sklearn.preprocessing import QuantileTransformer



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

    def __init__(self, merge_function=np.mean):
        self.merge_function = merge_function

    def __call__(self, sample: np.ndarray):
        return self.merge_function(sample, axis=(1, 2))


def load_data(directory: str, file_path: str, istrain,args):

    datalist = []
    masklist = []
    aug_datalist = []
    aug_masklist = []
    aug_labellist = []

    if istrain:
        labels = load_gt(file_path,args)

    all_files = np.array(
        sorted(
            glob(os.path.join(directory, "*.npz")),
            key=lambda x: int(os.path.basename(x).replace(".npz", "")),
        )
    )
    # in debug mode, only consider first 100 patches
    if args.debug:
        all_files = all_files[:10]
        if istrain: labels=labels[:10]

    for idx, file_name in enumerate(all_files):
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

    labels = gt_file[["P", "K", "Mg", "pH"]].values / np.array(
        [325.0, 625.0, 400.0, 7.8])  # normalize ground-truth between 0-1

    return labels


def preprocess(data_list, mask_list):
    def _shape_pad(data):
        max_edge = np.max(image.shape[1:])
        shape = (max_edge, max_edge)
        padded = np.pad(data, ((0, 0),
                               (0, (shape[0] - data.shape[1])),
                               (0, (shape[1] - data.shape[2]))),
                        'wrap')
        # print(padded.shape)
        return padded

    filtering = SpectralCurveFiltering()
    w1 = pywt.Wavelet('sym3')
    w2 = pywt.Wavelet('dmey')

    processed_data = []

    for idx, (data, mask) in enumerate(tqdm(zip(data_list, mask_list), total=len(data_list), position=0, leave=True,
                                            desc="INFO: Preprocessing data ...")):
        data = data / 1  # 2210  ## max-max=5419 mean-max=2210
        m = (1 - mask.astype(int))
        image = (data * m)

        image = _shape_pad(image)

        s = np.linalg.svd(image, full_matrices=False, compute_uv=False)
        s0 = s[:, 0] / 1  # np.max(s[:,0])
        s1 = s[:, 1] / 1  # np.max(s[:,1])
        s2 = s[:, 2] / 1  # np.max(s[:,2])
        s3 = s[:, 3] / 1  # np.max(s[:,3])
        s4 = s[:, 4] / 1  # np.max(s[:,4])
        dXds1 = s0 / s1
        # dXds2 = s1 /s2

        # s = s /np.expand_dims(np.linalg.norm(s,axis=0),0)
        # f=np.reshape(image,(150,image.shape[-1]*image.shape[-2]))
        # model = NMF(n_components=6, init='random', random_state=0,max_iter=1000)
        # model=FastICA(n_components=6, random_state=0,max_iter=1000, tol=0.01)
        # model=FactorAnalysis(n_components=6, random_state=0)
        # w = model.fit_transform(f)
        # model = NMF(n_components=3, init='random', random_state=0)
        # W = model.fit_transform(X)

        data = np.ma.MaskedArray(data, mask)
        arr = filtering(data)
        # arr = arr / np.max(arr)

        cA0, cD0 = pywt.dwt(arr, wavelet=w2, mode='constant')
        cAx, cDx = pywt.dwt(cA0[12:92], wavelet=w2, mode='constant')
        cAy, cDy = pywt.dwt(cAx[15:55], wavelet=w2, mode='constant')
        cAz, cDz = pywt.dwt(cAy[15:35], wavelet=w2, mode='constant')
        cAw2 = np.concatenate((cA0[12:92], cAx[15:55], cAy[15:35], cAz[15:25]), -1)
        cDw2 = np.concatenate((cD0[12:92], cDx[15:55], cDy[15:35], cDz[15:25]), -1)

        #cA0, cD0 = pywt.dwt(arr, wavelet=w1,mode='constant')
        #cAx, cDx = pywt.dwt(cA0[1:-1], wavelet=w1,mode='constant')
        #cAy, cDy = pywt.dwt(cAx[1:-1], wavelet=w1,mode='constant')
        #cAz, cDz = pywt.dwt(cAy[1:-1], wavelet=w1,mode='constant')
        #cAw1=np.concatenate((cA0,cAx,cAy, cAz),-1)
        #cDw1=np.concatenate((cD0,cDx,cDy, cDz),-1)

        dXdl = np.gradient(arr, axis=0)
        # dXdl = dXdl / np.max(dXdl)

        d2Xdl2 = np.gradient(dXdl, axis=0)
        # d2Xdl2 = d2Xdl2 / np.max(d2Xdl2)

        d3Xdl3 = np.gradient(d2Xdl2, axis=0)
        # d2Xdl2 = d2Xdl2 / np.max(d2Xdl2)
        # d4Xdl4 = np.gradient(d3Xdl3, axis=0)

        #fft = np.fft.fft(arr)
        #real = np.real(fft)
        # real = real / np.max(real)
        #imag = np.imag(fft)
        # imag = imag / np.max(imag)
        ffts = np.fft.fft(s0)
        reals = np.real(ffts)
        # real = real / np.max(real)
        imags = np.imag(ffts)
        # imag = imag / np.max(imag)

        #cos = dct(arr)

        #out = np.concatenate([arr, dXdl, d2Xdl2, d3Xdl3, dXds1, s0, s1, s2, s3, s4, real, imag, reals, imags, cDw1, cAw1, cDw2, cAw2,cos], -1)
        out = np.concatenate([arr, dXdl, d2Xdl2, d3Xdl3, dXds1, s0, s1, s2, s3, s4, reals, imags, cDw2, cAw2], -1)
        processed_data.append(out)

    return np.array(processed_data)


def mixing_augmentation(X, y, fract=0.1):
    mix_const = 0.05
    mix_index_1 = np.random.randint(X.shape[0], size=int(np.floor(X.shape[0] * fract)))
    mix_index_2 = np.random.randint(X.shape[0], size=int(np.floor(X.shape[0] * fract)))

    ma_X = (1 - mix_const) * X[mix_index_1] + mix_const * (X[mix_index_2])
    ma_y = (1 - mix_const) * y[mix_index_1] + mix_const * (y[mix_index_2])

    return np.concatenate([X, ma_X], 0), np.concatenate([y, ma_y], 0)


def evaluation_score(args, y_v, y_hat, y_b, cons):
    score = 0
    for i in range(len(args.col_ix)):
        print(f'Soil idx {i} / {len(args.col_ix) - 1}')
        mse_rf = mean_squared_error(y_v[:, i] * cons[i], y_hat[:, i] * cons[i])
        mse_bl = mean_squared_error(y_v[:, i] * cons[i], y_b[:, i] * cons[i])

        score += mse_rf / mse_bl

        print(f'Baseline MSE:      {mse_bl:.2f}')
        print(f'Random Forest MSE: {mse_rf:.2f} ({1e2 * (mse_rf - mse_bl) / mse_bl:+.2f} %)')
        print(f'Evaluation score: {score / len(args.col_ix)}')

    return score / 4



def predictions_and_submission(study, X_processed, X_test, y_train_col, cons, args):
    final_model = study.best_params["regressor"]
    if final_model == "RandomForest":
        # fit rf with best parameters on entire training data
        optimised_model = RandomForestRegressor(n_estimators=study.best_params['n_estimators'],
                                                max_depth=study.best_params['max_depth'],
                                                min_samples_leaf=study.best_params['min_samples_leaf'],
                                                n_jobs=-1)
    else:
        optimised_model = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror',
                                                                n_estimators=study.best_params['n_estimators'],
                                                                verbosity=1))

    optimised_model.fit(X_processed, y_train_col)
    predictions = optimised_model.predict(X_test)
    predictions = predictions * np.array(cons[:len(args.col_ix)])

    # calculate score on full training set
    baseline = BaselineRegressor()
    baseline.fit(X_processed, y_train_col)
    y_b = baseline.predict(X_processed)
    y_fulltrain_pred = optimised_model.predict(X_processed)

    score = evaluation_score(args, y_train_col, y_fulltrain_pred, y_b, cons)
    print(f'\nScore of best model ({final_model}) on training set: {score}\n')

    # print feature importances
    feats = {}
    importances = optimised_model.feature_importances_
    feature_names = ['arr', 'dXdl', 'd2Xdl2', 'd3Xdl3', 'dXds1', 's_0', 's_1', 's_2', 's_3', 's_4', 'real', 'imag',
                     'reals','imags', 'cDw1', 'cAw1', 'cDw2', 'cAw2', 'cos']
    for feature, importance in zip(feature_names, importances):
        feats[feature] = importance
    feats = sorted(feats.items(), key=lambda x: x[1], reverse=True)
    for feat in feats:
        print(f'{feat[0]}: {feat[1]}')

    # save the model
    if args.save_model:
        output_file = os.path.join(args.model_dir, f"{final_model}_{date_time}_"f"nest={study.best_params['n_estimators']}_maxd={study.best_params['max_depth']}_" \
                                                   f"minsl={study.best_params['min_samples_leaf']}_"f"aug_con={study.best_params['augment_constant']}_"f"aug_par={study.best_params['augment_partition']}.bin")

        with open(output_file, "wb") as f_out:
            joblib.dump(optimised_model, f_out)

    # only make submission file, if all 4 soil parameters are considered
    if len(args.col_ix) == 4 and args.debug == False:
        submission = pd.DataFrame(data=predictions, columns=["P", "K", "Mg", "pH"])
        print(submission.head())
        submission.to_csv(os.path.join(args.submission_dir, f"submission_{study.best_params['reg_name']}_" \
                                                            f"{date_time}_nest={study.best_params['n_estimators']}_maxd={study.best_params['max_depth']}_" \
                                                            f"minsl={study.best_params['min_samples_leaf']}_"f"aug_con={study.best_params['augment_constant']}_"f"aug_par={study.best_params['augment_partition']}.csv"),
                          index_label="sample_index")
        return predictions, submission

    return predictions


def predictions_and_submission_2(study, best_model, X_test, cons, args,min_score,label_transformer):

    predictions = []
    for rf in best_model:
        pp = rf.predict(X_test)
        if label_transformer is not None:
            pp=label_transformer.inverse_transform(pp)
        predictions.append(pp)

    predictions = np.asarray(predictions)

    predictions = np.mean(predictions, axis=0)
    predictions = predictions * np.array(cons[:len(args.col_ix)])


    feats = {}
    importances = best_model[-1].feature_importances_
    #feature_names = ['arr', 'dXdl', 'd2Xdl2', 'd3Xdl3', 'dXds1', 's_0', 's_1', 's_2', 's_3', 's_4', 'real', 'imag','reals', 'imags', 'cDw1', 'cAw1', 'cDw2', 'cAw2', 'cos']
    feature_names = ['arr', 'dXdl', 'd2Xdl2', 'd3Xdl3', 'dXds1', 's_0', 's_1', 's_2', 's_3', 's_4','reals', 'imags', 'cDw2', 'cAw2']

    for feature, importance in zip(feature_names, importances):
        feats[feature] = importance
    feats = sorted(feats.items(), key=lambda x: x[1], reverse=True)
    for feat in feats:
        print(f'{feat[0]}: {feat[1]}')

    submission = pd.DataFrame(data=predictions, columns=["P", "K", "Mg", "pH"])
    print(submission.head())
    if study is not None:
        submission.to_csv(os.path.join(args.submission_dir, f"submission_{study.best_params['regressor']}_" \
                                                            f"{date_time}_nest={study.best_params['n_estimators']}_maxd={study.best_params['max_depth']}_" \
                                                            f"minsl={study.best_params['min_samples_leaf']}_"f"aug_con={study.best_params['augment_constant']}_"f"aug_par={study.best_params['augment_partition']}.csv"),index_label="sample_index")
    else:
        submission.to_csv(os.path.join(args.submission_dir, "submission_best_{}.csv".format(min_score)),index_label="sample_index")


def main(args):
    train_data = os.path.join(args.in_data, "train_data", "train_data")
    test_data = os.path.join(args.in_data, "test_data")
    train_gt=os.path.join(args.in_data, "train_data", "train_gt.csv")

    # load the data
    print("start loading data ...")
    start_time = time.time()
    X_train, M_train, y_train, X_aug_train, M_aug_train, y_aug_train = load_data(train_data,train_gt, True, args)
    print(f"loading train data took {time.time() - start_time:.2f}s")
    print(f"train data size: {len(X_train)}")
    if args.debug == False:
        print(f"patch size examples: {X_train[0].shape}, {X_train[500].shape}, {X_train[1000].shape}")

    start_time = time.time()
    #y_train = load_gt(os.path.join(args.in_data, "train_data", "train_gt.csv"), args)
    X_test, M_test = load_data(test_data, None, False,args)

    print(f"loading test data took {time.time() - start_time:.2f}s")
    print(f"test data size: {len(X_test)}\n")

    print('Preprocess training data...')
    X_train = preprocess(X_train, M_train)
    X_aug_train = preprocess(X_aug_train, M_aug_train)

    print('preprocess test data ...')
    X_test = preprocess(X_test, M_test)




    # selected set of labels
    y_train_col = y_train[:, args.col_ix]
    y_aug_train_col=y_aug_train[:, args.col_ix]

    cons = np.array([325.0, 625.0, 400.0, 7.8])

    global best_model
    best_model = None
    global min_score
    min_score = 10
    global X_test_normalized_best
    global label_transformer_best
    def objective(trial):
        global best_model
        global min_score
        global X_test_normalized_best
        global label_transformer_best

        print(f"\nTRIAL NUMBER: {trial.number}\n")
        # training
        kfold = KFold(n_splits=args.folds, shuffle=True, random_state=RANDOM_STATE)

        X_train_normalized = np.zeros(X_train.shape)
        X_aug_train_normalized=np.zeros(X_aug_train.shape)
        X_test_normalized = np.zeros(X_test.shape)

        # min_max_scaler_list = []
        # https://machinelearningmastery.com/power-transforms-with-scikit-learn/
        # reg_name = trial.suggest_categorical("regressor", ["RandomForest", "XGB"])
        reg_name = trial.suggest_categorical("regressor", ["RandomForest"])

        if reg_name == "RandomForest":
            n_estimators = trial.suggest_int('n_estimators', args.n_estimators[0], args.n_estimators[1], log=True)
            max_depth = trial.suggest_categorical('max_depth', args.max_depth)
            min_samples_leaf = trial.suggest_categorical('min_samples_leaf', args.min_samples_leaf)

        else:
            n_estimators = trial.suggest_int('n_estimators', args.n_estimators[0], args.n_estimators[1], log=True)
            max_depth = trial.suggest_categorical('max_depth', args.max_depth)



        scaler_type = trial.suggest_categorical("scaler", ["robust", "minmax","None"])
        power_type = trial.suggest_categorical("power", ["yeo_johnson", "quantile","None"])
        label_trans_type = trial.suggest_categorical("label_trans", ["box_cox", "quantile", "None"])
        augment_constant = trial.suggest_int('augment_constant', 0, args.augment_constant, log=False)
        augment_partition = trial.suggest_int('augment_partition', args.augment_partition[0], args.augment_partition[1],log=True)

        for i in range(int(X_train.shape[-1] / 150)):
            if scaler_type=='robust': scaler = preprocessing.RobustScaler()
            elif scaler_type=='minmax': scaler = preprocessing.MinMaxScaler((0,1))
            else: scaler=None
            if scaler is not None:
                scaler.fit(np.concatenate((X_train[:, 150 * i:150 * i + 150], X_test[:, 150 * i:150 * i + 150])))
                X_train_normalized[:, 150 * i:150 * i + 150] = scaler.transform(X_train[:, 150 * i:150 * i + 150])
                X_aug_train_normalized[:, 150 * i:150 * i + 150] = scaler.transform(X_aug_train[:, 150 * i:150 * i + 150])
                X_test_normalized[:, 150 * i:150 * i + 150] = scaler.transform(X_test[:, 150 * i:150 * i + 150])
            else:
                X_train_normalized=np.array(X_train, copy=True)
                X_aug_train_normalized=np.array(X_aug_train, copy=True)
                X_test_normalized=np.array(X_test, copy=True)

            if power_type=='yeo_johnson': power = PowerTransformer(method='yeo-johnson')
            elif power_type == 'quantile':
                power = QuantileTransformer(n_quantiles=900, output_distribution="normal")
            else: power=None
            if power is not None:
                power.fit(np.concatenate((X_train_normalized[:, 150 * i:150 * i + 150], X_test_normalized[:, 150 * i:150 * i + 150])))
                X_train_normalized[:, 150 * i:150 * i + 150] = power.transform(X_train_normalized[:, 150 * i:150 * i + 150])
                X_aug_train_normalized[:, 150 * i:150 * i + 150] = power.transform(X_aug_train_normalized[:, 150 * i:150 * i + 150])
                X_test_normalized[:, 150 * i:150 * i + 150] = power.transform(X_test_normalized[:, 150 * i:150 * i + 150])

        if label_trans_type=='box_cox': label_transformer=PowerTransformer(method='box-cox')
        elif label_trans_type=='quantile': label_transformer=QuantileTransformer(n_quantiles=900, output_distribution="normal")
        else: label_transformer=None
        if label_transformer is not None:
            label_temp_list = [y_train_col]
            for idy in range(augment_constant):
                label_temp_list.append(y_aug_train_col[(idy + 1) * len(y_train) - augment_partition:(idy + 1) * len(y_train)])

            label_transformer.fit(np.concatenate(label_temp_list,axis=0))

        random_forests = []
        baseline_regressors = []
        y_hat_bl = []
        y_hat_rf = []
        scores = []

        print("START TRAINING ...")
        for i, (ix_train, ix_valid) in enumerate(kfold.split(np.arange(0, len(y_train)))):

            print(f'fold {i}:')

            X_t = X_train_normalized[ix_train]
            y_t = y_train_col[ix_train]


            for idy in range(augment_constant):
                X_ta_1 = X_aug_train_normalized[ix_train+(idy*len(y_train))]
                y_ta_1 = y_aug_train_col[ix_train+(idy*len(y_train))]
                X_t=np.concatenate((X_t,X_ta_1[-augment_partition:]),axis=0)
                y_t=np.concatenate((y_t,y_ta_1[-augment_partition:]),axis=0)

            # mixing augmentation
            if args.mix_aug:
                X_t, y_t = mixing_augmentation(X_t, y_t)

            X_v = X_train_normalized[ix_valid]
            y_v = y_train_col[ix_valid]

            # baseline
            baseline = BaselineRegressor()
            baseline.fit(X_t, y_t)
            baseline_regressors.append(baseline)

            #reg_name = trial.suggest_categorical("regressor", ["RandomForest", "XGB"])


            print(f"Training on {reg_name}")
            if reg_name == "RandomForest":
                # random forest
                model = RandomForestRegressor(n_estimators=n_estimators,
                                              max_depth=max_depth,
                                              min_samples_leaf=min_samples_leaf,
                                              n_jobs=-1)
            else:
                # xgboost
                model = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror',
                                                              max_depth=max_depth,
                                                              n_estimators=n_estimators,
                                                              verbosity=1))
            if label_transformer is not None:
                y_t=label_transformer.transform(y_t)
            model.fit(X_t, y_t)
            random_forests.append(model)
            #print(f'{reg_name} score: {model.score(X_v, y_v)}')

            # predictions
            y_hat = model.predict(X_v)
            if label_transformer is not None:
                y_hat = label_transformer.inverse_transform(y_hat)
            y_b = baseline.predict(X_v)

            y_hat_bl.append(y_b)
            y_hat_rf.append(y_hat)

            # evaluation score
            score = evaluation_score(args, y_v, y_hat, y_b, cons)
            scores.append(score)
            print(scores)

        print("END TRAINING")
        # final score
        mean_score = np.mean(np.array(scores))
        print(f'mean score: {mean_score}\n')
        if mean_score < min_score:
            min_score=mean_score
            best_model=random_forests
            label_transformer_best=label_transformer
            X_test_normalized_best=X_test_normalized
            predictions_and_submission_2(None, best_model, X_test_normalized, cons, args,min_score,label_transformer)

        return mean_score

    study = optuna.create_study(sampler=TPESampler(), direction='minimize')
    study.optimize(objective, n_trials=args.n_trials)
    predictions_and_submission_2(study, best_model, X_test_normalized_best, cons, args,min_score,label_transformer_best)

    # save study
    #final_model = study.best_params["regressor"]
    #if args.debug == False:
    #    output_file = os.path.join(args.submission_dir,
    #                               f"study_{final_model}_{date_time}_nest={study.best_params['n_estimators']}_maxd={study.best_params['max_depth']}_minsl={study.best_params['min_samples_leaf']}.pkl")
    #    with open(output_file, "wb") as f_out:
    #        joblib.dump(study, f_out)

    # prepare submission
    #print("MAKE PREDICTIONS AND PREPARE SUBMISSION")
    #print('preprocess test data ...')
    #X_test = preprocess(X_test, M_test)

    #if args.col_ix == 4:
    #    predictions, submission = predictions_and_submission(study, X_processed, X_test, y_train_col, cons, args)
    #else:
    #    predictions = predictions_and_submission(study, X_processed, X_test, y_train_col, cons, args)
    #print("PREDICTIONS AND SUBMISSION FINISHED")

    # save predictions
    #if args.save_pred:
    #    pass


if __name__ == "__main__":

    RANDOM_STATE = 42
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    now = datetime.now()
    date_time = now.strftime("%Y%m%d%H%M")

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--in-data', type=str, default='/p/project/hai_cons_ee/kuzu/ai4eo-hyperview/hyperview/keras/')
    parser.add_argument('--submission-dir', type=str,
                        default='/p/project/hai_cons_ee/kuzu/ai4eo-hyperview/hyperview/keras/modeldir/')
    parser.add_argument('--model-dir', type=str,
                        default='/p/project/hai_cons_ee/kuzu/ai4eo-hyperview/hyperview/keras/modeldir/')
    parser.add_argument('--save-pred', action='store_true', default=False)
    parser.add_argument('--save-model', action='store_true', default=False)
    parser.add_argument('--col-ix', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--mix-aug', action='store_true', default=False)
    # model hyperparams
    parser.add_argument('--n-estimators', type=int, nargs='+', default=[128, 1024])
    parser.add_argument('--max-depth', type=int, nargs='+', default=[4, 8, 16, 32, 64, 128, 256, None])
    parser.add_argument('--min-samples-leaf', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--n-trials', type=int, default=256)
    parser.add_argument('--augment-constant', type=int, default=5)
    parser.add_argument('--augment-partition', type=int, nargs='+', default=[100, 350])

    args = parser.parse_args()

    # output = os.path.join(args.submission_dir, f"out_{date_time}")
    # sys.stdout = open(output, 'w')

    print('BEGIN argparse key - value pairs')
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    print('END argparse key - value pairs')
    print()

    cols = ["P205", "K", "Mg", "pH"]

    main(args)

    # sys.stdout.close()
