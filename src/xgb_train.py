# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from new_dataset import load_data

import os
path = os.path.abspath(os.path.dirname(__file__))

# xgb 参数
params={
    "learning_rate":0.05,
    "max_depth":6,
    'n_estimators': 400,
    'min_child_weight': 1,
    'seed': 1030,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    # "objective":"multiclass",
    "objective": 'multi:softprob',
    "num_class":11,
    "silent":True,
    # 'tree_method':'gpu_hist'
}

# param1 = {'max_depth': 6, 'silent': 1, 'objective': 'multi:softprob', 'num_class': 3}
# params = {'learning_rate': 0.05, 'n_estimators': 400, 'max_depth': 6, 'min_child_weight': 1, 'seed': 0,
#           'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1, 'n_jobs': -1,
#           'num_class': 11, 'silent': 1, 'objective': 'multi:softprob'}



n_splits = 7
seed = 2018

def f1_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    preds = np.argmax(preds.reshape(15, -1), axis=0)
    score_vali = f1_score(y_true=labels, y_pred=preds, average='weighted')
    return 'f1_score', score_vali, True

def train_model(X, y, test):

    X_test = xgb.DMatrix(data=test)
    skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)

    xx_score, cv_pred = [], []

    for index,(train_index,test_index) in enumerate(skf.split(X,y)):

        print ("index: ", index)

        X_train, X_valid, y_train, y_valid = \
            X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        train_data = xgb.DMatrix(data=X_train, label=y_train)
        validation_data = xgb.DMatrix(data=X_valid, label=y_valid)

        print ("Training ....")
        # clf = xgb.train(params, train_data, num_boost_round=2000,
        #                 evals=[(train_data, 'train'), (validation_data, 'eval')], early_stopping_rounds=50,
        #                 feval=f1_score_vali, verbose_eval=1)
        clf = xgb.train(params, train_data, num_boost_round=2000,
                        evals=[(train_data, 'train'), (validation_data, 'eval')], early_stopping_rounds=50, verbose_eval=1)
        print ("Save model ...")
        clf.save_model(path + '/models/xgb_' + str(index))

        print ("Process X_valid ...")
        xx_pred = clf.predict(validation_data)
        xx_pred = np.argmax(xx_pred, axis=1)
        xx_score.append(f1_score(y_valid, xx_pred, average='weighted'))

        print ("Process X_test ...")
        y_test = clf.predict(X_test)
        y_test = np.argmax(y_test, axis=1)

        if index == 0:
            cv_pred = np.array(y_test).reshape(-1, 1)
        else:
            cv_pred = np.hstack((cv_pred, np.array(y_test).reshape(-1, 1)))

    print(xx_score, np.mean(xx_score))
    submit = []
    for line in cv_pred:
        submit.append(np.argmax(np.bincount(line)))
    return submit


if __name__ == '__main__':
    train, labels, test = load_data()

    service_label = [89950166, 89950167, 99999828, 89950168, 99999827, 99999826,\
                     99999830, 99999825, 90063345, 90109916,90155946]
    label_set = {k: i for i, k in enumerate(service_label)}
    for service in label_set.keys():
        labels.loc[labels['current_service'] == service, 'current_service'] = label_set[service]

    submit = train_model(train, labels, test)
    for i, v in enumerate(submit):
        submit[i] = service_label[v]
    data = pd.read_csv('./data/submit_sample.csv')
    data['current_service'] = submit
    data['current_service'] = data['current_service'].astype(int)
    data.to_csv('./temp/new_xgb_submission1015-3.csv', index=False)
