"""
Contains all my neural network architectures
"""
from sklearn import tree, ensemble, linear_model, svm
from sklearn.utils.class_weight import compute_class_weight as ccw
import config
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.layers as L
import config
import pickle
from statsmodels.tsa.arima_model import ARIMA

"""
=========================================================
NORMAL MACHINE LEARNING MODELS
=========================================================
"""
class BaseModel():
    """ Base model for a lot of models to inherit methods from """
    def __init__(self):
        self.model = None
    def fit(self, X, y):
        return self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)
    def save_feature_importances(self, cols, path):
        pass
    def save(self, path):
        with open(path, 'wb') as modelname:
            pickle.dump(self.model, modelname)

class Arima(BaseModel):
    def __init__(self):
        self.model = ARIMA(series, order=(5,1,0))

    def fit():
        self.model.fit()


class DecisionTree(BaseModel):
    def __init__(self):
        self.model = tree.DecisionTreeRegressor(random_state=42, max_depth=7)

    def save_feature_importances(self, X, path='../output/decision_tree_importances.png'):
        feature_imp = pd.DataFrame(sorted(zip(self.model.feature_importances_, X)), columns=['Value', 'Feature'])
        plt.figure(figsize=(20,10))
        sns.set(font_scale=2)
        img = sns.barplot(x='Value', y='Feature', data=feature_imp.sort_values(by='Value', ascending=False))
        plt.title('Decision Tree Features (avg over folds)')
        plt.tight_layout()
        img.get_figure().savefig(path)

    def save_decision_tree(self, features, target):
        fig = plt.figure(figsize=(25, 20))
        fig.set_size_inches(30, 20)
        _ = tree.plot_tree(self.model, feature_names=features, class_names=target, filled=True)
        fig.savefig('../output/decision_tree.png')

class RandomForest(BaseModel):
    def __init__(self):
        self.model = ensemble.RandomForestRegressor(random_state=42, bootstrap=True, criterion='mae',
                                                    max_depth=None, max_features='auto', max_leaf_nodes=None,
                                                    min_samples_leaf=1, n_estimators=100, verbose=0)
    def save_feature_importances(self, X, path):
        feature_imp = pd.DataFrame(sorted(zip(self.model.feature_importances_, X)), columns=['Value', 'Feature'])
        plt.figure(figsize=(20,10))
        sns.set(font_scale=2)
        img = sns.barplot(x='Value', y='Feature', data=feature_imp.sort_values(by='Value', ascending=False))
        plt.title('Random Forest Features (avg over folds)')
        plt.tight_layout()
        img.get_figure().savefig(path)

class LR(BaseModel):
    def __init__(self):
        self.model = linear_model.LinearRegression()

    def save_feature_importances(self, X, path):
        feature_imp = pd.DataFrame(sorted(zip(self.model.coef_, X)), columns=['Value', 'Feature'])
        plt.figure(figsize=(20,10))
        sns.set(font_scale=2)
        img = sns.barplot(x='Value', y='Feature', data=feature_imp.sort_values(by='Value', ascending=False))
        plt.title('Linear Regression Features (avg over folds)')
        plt.tight_layout()
        img.get_figure().savefig(path)

class XGBoost(BaseModel):
    def __init__(self):
        self.model = xgb.XGBRegressor(num_leaves=2**7 + 1, max_depth=15, metric='mae', learning_rate=0.008,
                                      bagging_fraction=1, subsample=1, reg_alpha=0.2, reg_lambda=0.1,
                                      feature_fraction=0.5, random_state=42, importance_type='gain',
                                      bagging_frequency=6, bagging_seed=42, verbosity=0, max_bin=512,
                                      n_estimators=1000)

    def save_feature_importances(self, X, path='../output/rf_importances.png'):
        feature_imp = pd.DataFrame(sorted(zip(self.model.feature_importances_, X)), columns=['Value', 'Feature'])
        plt.figure(figsize=(20,10))
        sns.set(font_scale=2)
        img = sns.barplot(x='Value', y='Feature', data=feature_imp.sort_values(by='Value', ascending=False))
        plt.title('XGBoost Features (avg over folds)')
        plt.tight_layout()
        img.get_figure().savefig(path)

class SVM(BaseModel):
    def __init__(self):
        self.model = svm.SVR(C=1)

    def save_feature_importances(self, X, path):
        feature_imp = pd.DataFrame(sorted(zip(self.model.coef_, X)), columns=['Value', 'Feature'])
        plt.figure(figsize=(20,10))
        sns.set(font_scale=2)
        img = sns.barplot(x='Value', y='Feature', data=feature_imp.sort_values(by='Value', ascending=False))
        plt.title('SVM Features (avg over folds)')
        plt.tight_layout()
        img.get_figure().savefig(path)

class LGB(BaseModel):
    def __init__(self):
        # regression
        # self.model = lgb.LGBMRegressor(num_leaves=2**7 + 1, max_depth=15, metric='mae', learning_rate=0.008,
        #                                bagging_fraction=1, subsample=1, reg_alpha=0.2, reg_lambda=0.1,
        #                                feature_fraction=0.5, random_state=42, importance_type='gain',
        #                                bagging_frequency=6, bagging_seed=42, max_bin=512,
        #                                n_estimators=1000)
        # lgb_params = {'objective': 'multiclass', 'num_class': 11, 'metric': 'multi_logloss', 'learning_rate': 0.01,
        #               'lambda_l1': 0.001, 'lambda_l2': 0.18977, 'num_leaves': 180, 'feature_fraction': 0.587338,
        #               'bagging_fraction': 0.705783, 'bagging_freq': 4 }
        # classification
        self.model = None

    def fit(self, X, y):
        cw = ccw('balanced', np.unique(y), y)
        class_weights = {0:cw[0], 1:cw[1]}
        self.model = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', learning_rate=0.1,
                                        feature_fraction=0.9, reg_lambda=0.2, num_leaves=50,
                                        class_weight=class_weights, max_depth=7)
        return self.model.fit(X, y)

    def save_feature_importances(self, X, path='../output/lgb_importances.png'):
        feature_imp = pd.DataFrame(sorted(zip(self.model.feature_importances_, X)), columns=['Value', 'Feature'])
        plt.figure(figsize=(20,10))
        sns.set(font_scale=2)
        img = sns.barplot(x='Value', y='Feature', data=feature_imp.sort_values(by='Value', ascending=False))
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout()
        img.get_figure().savefig(path)

    def get_class_weight(classes, exp=1):
        '''
        Weight of each class will be used during the training process
        '''
        hist, _ = np.histogram(classes, bins=np.arange(12)-0.5)
        class_weight = hist.sum()/np.power(hist, exp)
        return class_weight

"""
====================================================
DEEP LEARNING MODELS
====================================================
"""
class MLP():
    def __init__(self):
        shape = 6
        self.X_input = L.Input(shape=(shape,))
        self.X = L.Dense(8, activation='relu')(self.X_input)
        self.X = L.Dense(16, activation='relu')(self.X)
        # self.X = L.Dense(8, activation='relu')(self.X)
        self.X = L.Dense(1)(self.X)

    def fit(self, X, y):
        adam = Adam(learning_rate=0.001)
        self.model = Model(inputs=self.X_input, outputs=self.X)
        self.model.compile(optimizer=adam, loss='mean_absolute_error', metrics=['mae'])
        return self.model.fit(x=X, y=y, epochs=50, batch_size=32, verbose=1, validation_split=0.2)

    def predict(self, X):
        self.preds = self.model.predict(X)
        return self.preds

    def save(self, path):
        return self.model.save(path)


class LSTM():
    def __init__(self):
        shape = 6
        self.X_input = L.Input((shape, 1))
        self.X = L.LSTM(50, return_sequences=True, activation='relu')(self.X_input)
        self.X = L.LSTM(50, return_sequences=False, activation='relu')(self.X)
        self.X = L.Dense(16, activation='relu')(self.X)
        self.X = L.Dense(1)(self.X)

    def fit(self, X, y):
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        adam = Adam(learning_rate=0.001)
        # compile and fit the model
        self.model = Model(inputs=self.X_input, outputs=self.X)
        self.model.compile(optimizer=adam, loss='mean_absolute_error', metrics=['mae'])
        return self.model.fit(x=X, y=y, epochs=100, batch_size=32, verbose=1, validation_split=0.2)

    def predict(self, X):
        # reshaping x_valid to be used for prediction in lstm
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        self.preds = self.model.predict(X)
        self.preds = np.squeeze(self.preds)
        return self.preds

    def save(self, path):
        return self.model.save(path)

    def save_feature_importances(self, cols, path):
        pass