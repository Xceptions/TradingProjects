Start by creating folds, then use your train.py to keep testing models with features,
When you generate a good feature, run it against another model, iterate until you are
confident in your models and features
When you have gotten a good feature set, save them using the --save flag
Then use the inference.py to build the predictions for different models
The models will be saved in models using their modelname and fold number
The predictions will be save in output as train_preds.csv and test_preds.csv
ensemble.py will fetch the train and test predictions and ensemble them

neuralnetwork.py is for performing deep learning on the data


train.py -> inference.py -> neuralnetwork -> ensemble.py