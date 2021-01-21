import numpy as np
import pandas as pd
import config


def calc_perc_change(arg):
    x = arg[0]
    y = arg[1]
    ans = ((x - y) / y) * 100
    return ans

def run(files):
    for j in files:
        dt = pd.read_csv(f'../output/{j}.csv')

        for i in ['SVM', 'LGB', 'close_tomorrow']:
            dt[f'{i} %change'] = dt[[i, 'close']].apply(calc_perc_change, axis=1)
        dt.to_csv(f'{config.INFERENCE_OUTPUT}{j}_perc_check.csv', index=False)
    print("done")

    

run(['v1_train_inference', 'v1_test_inference'])
