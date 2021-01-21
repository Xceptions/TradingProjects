import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as mae
import xgboost as xgb

# build all forms of ensemble
def mean_predictions(preds):
    """ Return mean predictions """
    return np.mean(preds, axis=1)

def rank_mean():
    pass

def run():
    target = 'close_tomorrow'
    train = pd.read_csv('../output/v3_train_inference.csv')
    test = pd.read_csv('../output/v3_test_inference.csv')

    train_y = train[target]
    test_y = test[target]
    train.drop([target], axis='columns', inplace=True)
    test.drop([target], axis='columns', inplace=True)

    lr = LinearRegression()
    lr.fit(train, train_y)
    preds = lr.predict(test)
    print(mae(test_y, preds))
    
    # to view your predictions.
    pred_df = pd.DataFrame()
    pred_df['True'] = test_y
    pred_df['preds'] = preds
    # pred_df.to_csv(f'../output/output_{fold}.csv', index=False)
    print(pred_df.head(10))

if __name__ == "__main__":
    run()

