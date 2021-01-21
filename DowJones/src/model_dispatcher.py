import config
import models

def huber_approx_obj(preds, dtrain):
    '''
    xgboost optimizing function for mean absolute error
    '''
    d = preds - dtrain #add .get_labels() for xgb.train()
    h = 1  #h is delta in the graphic
    scale = 1 + (d / h) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt
    hess = 1 / scale / scale_sqrt
    return grad, hess

def MacroF1Metric():
    """
    LGB Params used to train this loss
        lgb_params = {'objective': 'multiclass', 'num_class': 11, 'metric': 'multi_logloss', 'learning_rate': 0.01,
                      'lambda_l1': 0.001, 'lambda_l2': 0.18977, 'num_leaves': 180, 'feature_fraction': 0.587338,
                      'bagging_fraction': 0.705783, 'bagging_freq': 4 }
    """
    labels = dtrain.get_label()
    num_labels = 11
    preds = preds.reshape(num_labels, len(preds)//num_labels)
    preds = np.argmax(preds, axis=0)
    score = f1_score(labels, preds, average="macro")
    return ('KaggleMetric', score, True)


models = {
    "dt": models.DecisionTree(),
    "rf": models.RandomForest(),
    "lr": models.LR(),
    "xgb": models.XGBoost(),
    "svm": models.SVM(),
    "lgb": models.LGB(),
    "mlp": models.MLP(),
    "lstm": models.LSTM()
}


# to get the final accuracy, calculate the mean and the mean absolute error should be the percentage of the
# performance since he wants to see performance
