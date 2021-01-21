"""
========================================
CREATE VARIOUS FOLDS OF OUR DATA
========================================
"""

import pandas as pd
from sklearn import model_selection


if __name__ == "__main__":
    # read train data
    df = pd.read_csv("../input/train.csv")

    # create a new column called kfold and fill it with -1
    df["kfold"] = -1

    # # randomize the rows of the data
    # df = df.sample(frac=1).reset_index(drop=True)

    # initialize the kfold class
    # kf = model_selection.TimeSeriesSplit(n_splits=5)
    # For stratified
    kf = model_selection.StratifiedKFold(n_splits=5)
    y = df.return_class.values

    # fill the new kfold column
    for fold, (t_, v_) in enumerate(kf.split(X=df, y=y)): # for stratified: kf.split(X=df, y=y)
        df.loc[v_, 'kfold'] = fold

    # save the new csv with kfold column
    df.to_csv("../input/train_class_folds.csv", index=False)

"""
1 -------------> Price Fall
2 -------------> Price Stagnant
3 -------------> Price Rise
"""