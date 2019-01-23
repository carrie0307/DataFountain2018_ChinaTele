# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from new_dataset import load_data
#
X_train, y_train, test = load_data()
feature_names = X_train.columns

# X_train = pd.read_csv('./data/train.csv', low_memory=False, index_col='user_id')
# X_train = X_train[X_train['current_service'] != 89016252]
# X_train = X_train[X_train['current_service'] != 89016259]
# X_train = X_train[X_train['current_service'] != 89016253]
# X_train = X_train[X_train['current_service'] != 99104722]
# X_train = X_train.replace('\\N', np.nan)
# y_train = X_train[['current_service']]


def randomforest_Classifier():

    # clf = RandomForestClassifier(n_estimators= 50, max_depth=10,criterion='gini', min_samples_split=5, max_features='auto',
    #                              bootstrap=True, random_state=2018, verbose=2, n_jobs=-1,min_samples_leaf=4)
    clf = RandomForestClassifier()
    print ("fitting ...")
    clf.fit(X_train, y_train.values.ravel())
    feature_evaluation = sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), feature_names), reverse=True)
    print (feature_evaluation)


def PCA_evaluation():
    pca = PCA(n_components=None)
    pca.fit(X_train)
    # print(pca.explained_variance_)
    variance = sorted(zip(map(lambda x: round(x, 4), pca.explained_variance_), feature_names), reverse=True)
    for item in variance:
        print (item)
    print ("\n")
    # 返回 所保留的n个成分各自的方差百分比。
    variance_ratio = sorted(zip(map(lambda x: round(x, 4), pca.explained_variance_ratio_), feature_names), reverse=True)
    for item in variance_ratio:
        print (item)

def RFE_evaluation():
    """
    给定一个外部的estimator，为feature分配权重（例如：线性模型的相关系数coefficients），递归特征淘汰（RFE）通过递归将feature集越来越小。
    recursive feature elimination(RFE) is to select features by recursively considering smaller and smaller sets of features.
    """
    pass






if __name__ == '__main__':
    randomforest_Classifier()
    # PCA_evaluation()