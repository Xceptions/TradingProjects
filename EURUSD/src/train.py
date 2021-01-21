"""
=================================================================
TRAIN OUR VARIOUS FOLDS USING MODELS SUPPLIED BY model_dispatcher
=================================================================
"""

import os
import argparse
import random

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics, tree
from sklearn.preprocessing import StandardScaler

# our modules
import config
import model_dispatcher
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')


train_stats = {}

def engineer_features(df, datatype='train'):
    # if test, perform feature inferencing e.g. apply mean of train

    def calc_roll_stats(s, windows=[3, 5, 30]):
        roll_stats = pd.DataFrame()
        for w in windows:
            roll_stats['n_roll_mean_' + str(w)] = s.shift(1).rolling(window=w, min_periods=1).mean()
            roll_stats['n_roll_max_' + str(w)] = s.shift(1).rolling(window=w, min_periods=1).max()
            roll_stats['n_roll_min_' + str(w)] = s.shift(1).rolling(window=w, min_periods=1).min()
        roll_stats = roll_stats.fillna(value=0)
        return roll_stats

    df['close_tomorrow'] = df['Close'].shift(periods=-1)
    df['close_yesterday'] = df['Close'].shift(periods=1)
    rollstats = calc_roll_stats(df['Close'])
    df = df.join(rollstats)
    df = df[['Close', 'close_yesterday', 'close_tomorrow', 'n_roll_mean_5',
             'n_roll_mean_30', 'n_roll_max_5', 'n_roll_max_30', 'n_roll_min_5',
             'n_roll_min_30', 'n_roll_mean_3', 'n_roll_max_3', 'n_roll_min_3']]
    df.dropna(inplace=True)

    return df

def save_features(name):
    df_train = pd.read_csv(config.TRAINING_FILE)
    df_test = pd.read_csv(config.TESTING_FILE)

    df_train = engineer_features(df_train, datatype='train')
    df_test = engineer_features(df_test, datatype='valid')

    df_train.to_csv(f'{config.PROCESSED_INPUT}{name}_train.csv', index=False)
    df_test.to_csv(f'{config.PROCESSED_INPUT}{name}_test.csv', index=False)

def run(fold, model, save):
    df = pd.read_csv(config.TRAINING_FILE_FOLDS)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    """ ======================= Feature engineering ==================== """
    df_train = engineer_features(df_train, datatype='train')
    df_valid = engineer_features(df_valid, datatype='valid')
    """ ================================================================ """

    # drop the label column from dataframe and convert it to
    # a numpy array
    features = [x for x in df_train.columns if x != 'close_tomorrow']
    # features = ['Close']
    
    x_train = df_train[features].values
    y_train = df_train['close_tomorrow'].values

    x_valid = df_valid[features].values
    y_valid = df_valid['close_tomorrow'].values

    # fetch the model from model_dispatcher
    clf = model_dispatcher.models[model]

    print(f'Training features: {features}')
    clf.fit(x_train, y_train)

    preds = clf.predict(x_valid)
    clf.save_feature_importances(features) # run function after you are done with the feature engineering

    """
    # to view your predictions.
    pred_df = pd.DataFrame()
    pred_df['y_valid'] = y_valid
    pred_df['preds'] = preds
    pred_df.to_csv(f'../output/output_{fold}.csv', index=False)
    """

    mae = metrics.mean_absolute_error(y_valid, preds)
    print(f"Fold={fold}, MAE={mae}")
    
    # save the features engineered if option is provided with name
    if save != 'None':
        save_features(save)

    # save the model
    # joblib.dump(clf,
    #             os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin")
    # )
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--save", type=str, default='None')
    args = parser.parse_args()

    if args.fold == "all":
        for x in range(5):
            run(fold=x, model=args.model, save=args.save)
    else:
        fold = int(args.fold)
        run(fold=fold, model=args.model, save=args.save)


# to run: 
# python train.py --fold 0 --model decision_tree_gini --save v1
# python train.py --fold all
# to ignore warning: python -W ignore train.py
# you can create a shell script to run all folds