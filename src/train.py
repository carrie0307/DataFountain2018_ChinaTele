# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn import svm,tree
from sklearn.metrics import accuracy_score,f1_score
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from sklearn.decomposition import PCA
from dataset import load_data

X_train, X_val, y_train, y_val, test = load_data()

def randomforest_Classifier():

    # clf = RandomForestClassifier(n_estimators= 50, max_depth=10,criterion='gini', min_samples_split=5, max_features='auto',
    #                              bootstrap=True, random_state=2018, verbose=2, n_jobs=-1,min_samples_leaf=4)
    clf = RandomForestClassifier()
    print ("fitting ...")
    clf.fit(X_train, y_train.values.ravel())
    print ("predicting ... ")
    result = clf.predict(test)
    data = pd.read_csv('./data/submit_sample.csv')
    data['current_service'] = result

    data.to_csv('./result/RandomForest_submission.csv', index=False)
    # get dummy 87, oneencoder 89
    X_val_pred = clf.predict(X_val)
    print (f1_score(y_val, X_val_pred,average='weighted'))


def svm_Classlifer():

    clf = svm.SVC(verbose=True)
    print ("fitting ...")
    clf.fit(X_train, y_train.values.ravel())
    print ("fitted ...")
    result = clf.predict(test)
    print (result)
    data = pd.read_csv('./data/submit_sample.csv')
    data['predict'] = result
    data.to_csv('./result/svm_submission.csv', index=False)


def DTree_Classlifer():

    # clf = tree.DecisionTreeClassifier(criterion='gini', splitter='random', max_depth=None, min_samples_split=5, min_samples_leaf=5,
    #                                   min_weight_fraction_leaf=0.0, max_features=None, random_state=42, max_leaf_nodes=None,
    #                                   min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)
    print ("DecisionTree ... ")
    clf = tree.DecisionTreeClassifier()
    print ("fitting ...")
    clf.fit(X_train, y_train.values.ravel())
    print ("fitted ...")
    result = clf.predict(test)
    # print (result)
    data = pd.read_csv('./data/submit_sample.csv')
    data['current_service'] = result
    # data.to_csv('./result/decisiontree_submission.csv', index=False)
    data.to_csv('./result/decisiontree_submission.csv', index=True)
    print (clf.score(X_val, y_val))
    # 输出各个特征项的重要指数
    # print (clf.feature_importances_)
    # 输出各个参数
    # print (clf.get_params())
    X_val_pred = clf.predict(X_val)
    print(f1_score(y_val, X_val_pred, average='weighted'))

def Adaboost_Classlifer():

    clf = AdaBoostClassifier(n_estimators=100,random_state=42)
    print ("fitting ...")
    clf.fit(X_train, y_train.values.ravel())
    print ("fitted ...")
    result = clf.predict(test)
    # print (result)
    data = pd.read_csv('./data/submit_sample.csv')
    data['predict'] = result
    data.to_csv('./result/adaboosting_submission.csv', index=False)
    print (clf.score(X_val, y_val))

def xgb_Classsifer():

    print ("X_train: ", type(X_train), "y_train: ", type(y_train))
    print("X_val: ", type(X_val), "y_val: ", type(y_val))
    print ("test: ", type(test))
    train_data = xgb.DMatrix(data=X_train, label=y_train.values.ravel())
    x_val_data = xgb.DMatrix(data=X_val)
    test_data = xgb.DMatrix(data=test)
    param = {'silent': 0, 'eval_metric': 'rmse'}

    print("fitting ...")
    model = xgb.train(param, train_data)
    # clf.fit(X_train,y_train)
    print("fitted ...")

    # 对测试数据进行预测
    result = model.predict(test_data)
    data = pd.read_csv('./data/submit_sample.csv')
    data['predict'] = result
    data.to_csv('./result/xgb_submission.csv', index=False)

    test_predicted = model.predict(x_val_data)
    # 用训练集划分出的测试数据计算准确率
    predictions = [round(value) for value in test_predicted]
    print (predictions)
    accuracy = accuracy_score(y_val, predictions)
    print (accuracy)

def gboost_Classifier():

    clf = GradientBoostingClassifier(random_state=42,verbose = 1,n_estimators=200,max_depth=4)
    print("fitting ...")
    clf.fit(X_train, y_train.values.ravel())
    print("fitted ...")
    result = clf.predict(test)
    # print (result)
    data = pd.read_csv('./data/submit_sample.csv')
    data['predict'] = result
    data.to_csv('./result/gboost_submission.csv', index=False)
    print(clf.score(X_val, y_val))
    # 86.84
    # 输出各个特征项的重要指数
    print(clf.feature_importances_)
    # 输出各个参数 f
    print(clf.get_params())



if __name__ == '__main__':
    # randomforest_Classifier()
    # svm_Classlifer()
    DTree_Classlifer()
    # Adaboost_Classlifer()
    """XGB文件异常"""
    # xgb_Classsifer()
    # gboost_Classifier()
