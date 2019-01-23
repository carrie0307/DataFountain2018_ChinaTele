
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from new_dataset import load_data

import os
path = os.path.abspath(os.path.dirname(__file__))

fold = 7
# param1 = {'max_depth': 6, 'objective': 'multiclass', 'num_class': 3, "lambda_l1": 0.1, "lambda_l2": 0.2, 'n_jobs': -1}
# param4 = {'learning_rate': 0.05, 'max_depth': 6, 'min_child_weight': 1, 'seed': 0,
#           'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0, 'reg_lambda': 1, 'n_jobs': -1,
#           'num_class': 8, 'objective': 'multiclass', "lambda_l1": 0.1, "lambda_l2": 0.2}

params = {'learning_rate': 0.05, 'max_depth': 6, 'min_child_weight': 1, 'seed': 0,
          'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0, 'reg_lambda': 1, 'n_jobs': -1,
          'num_class': 11, 'objective': 'multiclass', "lambda_l1": 0.1, "lambda_l2": 0.2}

def f1_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    preds = np.argmax(preds.reshape(11, -1), axis=0)
    score_vali = f1_score(y_true=labels, y_pred=preds, average='weighted')
    return 'f1_score', score_vali, True


def train_model(X, y, test):

    X_test = test
    skf = StratifiedKFold(n_splits=fold, random_state=2018, shuffle=True)

    xx_score, cv_pred = [], []

    for index,(train_index,test_index) in enumerate(skf.split(X,y)):

        print ("index: ", index)

        X_train, X_valid, y_train, y_valid = \
            X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        train_data = lgb.Dataset(data=X_train, label=y_train)
        validation_data = lgb.Dataset(data=X_valid, label=y_valid)

        print ("Training ....")
        # clf = lgb.train(params, train_data, num_boost_round=3000, valid_sets=[validation_data],
        #                 early_stopping_rounds=50, feval=f1_score_vali, verbose_eval=1)
        clf = lgb.train(params, train_data, num_boost_round=3000, valid_sets=[validation_data],
                         early_stopping_rounds=50, feval=f1_score_vali, verbose_eval=1)
        print ("Save model ...")
        clf.save_model(path + '/models/lgb_' + str(index))

        print ("Process X_valid ...")
        xx_pred = clf.predict(X_valid)
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
    data.to_csv('./temp/lgb_new_submission1014.csv', index=False)