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

models = {
    "dt": models.DecisionTree(),
    "rf": models.RandomForest(),
    "lr": models.LR(),
    "xgb": models.XGBoost(),
    "svm": models.SVM(),
    "lgb": models.LGB(),
    # "mlp": models.MLP(),
    "lstm": models.LSTM()
}


# to get the final accuracy, calculate the mean and the mean absolute error should be the percentage of the
# performance since he wants to see performance
