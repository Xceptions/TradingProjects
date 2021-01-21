# sets seed for the whole project

def set_seed(seed):
    import numpy as np
    import tensorflow as tf
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(2020)


TRAINING_FILE_FOLDS = "../input/raw/train_folds.csv"
TRAINING_FILE = '../input/raw/train.csv'
TESTING_FILE = '../input/raw/test.csv'

PROCESSED_INPUT = '../input/processed/'
INFERENCE_OUTPUT = '../output/'

MODEL_OUTPUT = "../models/"