import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold, KFold

import scipy.stats as st
import model_dispatcher
import tensorflow as tf

import gc, pickle
import warnings
warnings.filterwarnings("ignore")
import argparse
import config


def transformer(y, func=None):
    if func is None:
        return y
    else:
        return func(y)

def stacking(models, X_train, y_train, X_test, y_test,
             regression=True, transform_target=None,
             transform_pred=None, metric=None,
             n_folds=4, stratified=False, shuffle=False,
             random_state=2020, verbose=0):
    # Print type of task
    if regression and verbose > 0:
        print('task: [regression]')
    elif not regression and verbose > 0:
        print('task: [classification]')
        
    # Specify default metric for cross-validation
    if metric is None and regression:
        metric = mean_absolute_error
    elif metric is None and not regression:
        metric = f1_score
    # Print metric
    if verbose > 0:
        print('metric: [%s]\n' % metric.__name__)
        
    # Split indices to get folds (stratified should only be used only for classification)
    if stratified and not regression:
        kf = StratifiedKFold(n_folds, shuffle=shuffle, random_state=random_state)
    else:
        kf = KFold(n_folds, shuffle=shuffle, random_state=random_state)
        
    # Create empty numpy arrays for stacking features
    S_train = np.zeros((X_train.shape[0], len(models)))
    # S_holdout = np.zeros((X_holdout.shape[0], len(models_and_features)))
    S_test = np.zeros((X_test.shape[0], len(models)))
    S_modelnames = []
    results = [] # where all the models and scores will be stored
    
    # Loop across models
    for model_counter, model in enumerate(models):
        model_name_ = model.__class__.__name__
        if verbose > 0:
            print('model %d: [%s]' % (model_counter, model_name_))
        S_modelnames.append(model_name_)
            
        # Create empty numpy array, which will contain temporary predictions for test set made in each fold
        # S_holdout_temp = np.zeros((X_holdout.shape[0], n_folds))
        S_test_temp = np.zeros((X_test.shape[0], n_folds))
        
        # Loop across folds
        for fold_counter, (tr_index, vl_index) in enumerate(kf.split(X_train, y_train)):
            X_tr = X_train.iloc[tr_index].values
            y_tr = y_train.iloc[tr_index].values
            X_vl = X_train.iloc[vl_index].values
            y_vl = y_train.iloc[vl_index].values
            X_te = X_test.values
            y_te = y_test.values
            
            # Fit 1-st level model
            model.fit(X_tr, transformer(y_tr, func = transform_target))

            # save the models
            model.save(f'{config.MODEL_OUTPUT}{model_name_}_{fold_counter}.h5')
            model.save_feature_importances(X_train.columns,
                                           f'{config.INFERENCE_OUTPUT}{model_name_}_{fold_counter}.jpg')

            # Predict out-of-fold part of train set
            S_train[vl_index, model_counter] = transformer(model.predict(X_vl), func=transform_pred)
            # Predict full test set
            pred = transformer(model.predict(X_te), func=transform_pred)
            S_test_temp[:, fold_counter] = pred
            
            if verbose > 1:
                print(' fold %d: [%.8f]' % (fold_counter, metric(y_vl, S_train[vl_index, model_counter])))
                
        # Compute mean or mode of predictions for test set
        if regression:
            S_test[:, model_counter] = np.mean(S_test_temp, axis=1)
        else:
            S_test[:, model_counter] = st.mode(S_test_temp, axis=1)[0].ravel()

        err_ = metric(y_train, S_train[:, model_counter])
        results.append([model_name_, err_])
        if verbose > 0:
            print('---------')
            print('MEAN: [%.8f]\n' % (err_))
            print('-----------------------------------------------------')

    pd.DataFrame(results).to_csv(f'{config.INFERENCE_OUTPUT}results.csv', index=False)     
    return (S_train, S_test, S_modelnames)


def run(train, test, target, name):
    '''using the stacking ensembler'''

    tr_cols = [x for x in train.columns if x != target]
    train_x = train[tr_cols]
    train_y = train[target]
    test_x = test[tr_cols]
    test_y = test[target]

    # get all models being used in the model_dispatcher file add them to a list
    models_1 = []
    for key, value in model_dispatcher.models.items():
        models_1.append(value)

    S_train, S_test, S_modelnames = stacking(models_1, train_x, train_y, test_x, test_y, regression=True,
                                             transform_target=None, transform_pred=None, metric=None,
                                             n_folds=2, stratified=False, shuffle=True, random_state=2020,
                                             verbose=5)

    train_preds = pd.DataFrame(data=S_train, columns=S_modelnames)
    train_preds[target] = train_y
    train_preds['close'] = train['Close']
    del S_train
    gc.collect()

    test_preds = pd.DataFrame(data=S_test, columns=S_modelnames)
    test_preds[target] = test_y
    test_preds['close'] = test['Close']
    del S_test
    gc.collect()

    train_preds.to_csv(f"{config.INFERENCE_OUTPUT}{name}_train_inference.csv", index=False)
    test_preds.to_csv(f"{config.INFERENCE_OUTPUT}{name}_test_inference.csv", index=False)
    del train_preds, test_preds
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str)
    args = parser.parse_args()

    train = pd.read_csv(f'{config.PROCESSED_INPUT}{args.name}_train.csv')
    test = pd.read_csv(f'{config.PROCESSED_INPUT}{args.name}_test.csv')

    run(train, test, 'close_tomorrow', args.name)

# to run: python inference.py --name v1
# produces models and predictions saved in output