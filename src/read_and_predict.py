
"""
 load the model saved and predict the test

"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
import lightgbm as lgb
from new_dataset import load_data

import os
path = os.path.abspath(os.path.dirname(__file__))

service_label = [89950166, 89950167, 99999828, 89950168, 99999827, 99999826,\
                     99999830, 99999825, 90063345, 90109916,90155946]

params = {
        "learning_rate": 0.1,
        "lambda_l1": 0.1,
        "lambda_l2": 0.2,
        "max_depth": 4,
        # "objective":"multiclass",
        "objective": 'multi:softprob',
        "num_class": 11,
        "silent": True,
    }



# def load_model(model_filename):
#     model_path = path + '/models/' + model_filename
#     model = load_model(model_path)

def xgb_predict(model_filename, test):

    xgb_model = xgb.Booster(params)
    print ("Loading Model ...")
    xgb_model.load_model(path + '/models/' + model_filename)
    X_test = xgb.DMatrix(data=test)
    print ("Predicting ... ")
    submit = xgb_model.predict(X_test)
    submit = np.argmax(submit, axis=1)
    print ("Predict over ...")
    for i, v in enumerate(submit):
        submit[i] = service_label[v]
    data = pd.read_csv('./data/submit_sample.csv')
    data['current_service'] = submit
    data['current_service'] = data['current_service'].astype(int)
    data.to_csv('./temp/xgb_read_predict1017-.csv', index=False)

lgb_params = {'learning_rate': 0.05, 'max_depth': 6, 'min_child_weight': 1, 'seed': 0,
          'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0, 'reg_lambda': 1, 'n_jobs': -1,
          'num_class': 11, 'objective': 'multiclass', "lambda_l1": 0.1, "lambda_l2": 0.2}

def lgb_predict(model_filename, test):

    print (model_filename)
    lgb_model = lgb.Booster(model_file=path + '/models/' + model_filename)
    print("Loading Model ...")
    print ("Predicting ... ")
    submit = lgb_model.predict(test)
    submit = np.argmax(submit, axis=1)
    print ("Predict over ...")
    for i, v in enumerate(submit):
        submit[i] = service_label[v]
    data = pd.read_csv('./data/submit_sample.csv')
    data['current_service'] = submit
    data['current_service'] = data['current_service'].astype(int)
    data.to_csv('./temp/lgb_prediction-new.csv', index=False)

if __name__ == '__main__':
    train, labels, test = load_data()
    # lgb_predict('lgb_2', test)
    xgb_predict('xgb_3', test)
