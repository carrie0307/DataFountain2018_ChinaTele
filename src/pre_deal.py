# -*- coding: utf-8 -*-

'''

  主要是一些测试用代码了
  
'''
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,Imputer
from sklearn.decomposition import PCA
# from reportgen.utils.preprocessing import Discretization
from sklearn.preprocessing import KBinsDiscretizer

# feature:  gender
# 1     400572
# 2     185594
# 0      18058
# 01      5476
# 02      2729
# 00       221
# \N         2
# Name: gender, dtype: int64

def status(x):
    return pd.Series([x.count(),x.min(),x.idxmin(),x.quantile(.25),x.median(),
                      x.quantile(.75),x.mean(),x.max(),x.idxmax(),x.mad(),x.var(),
                      x.std(),x.skew(),x.kurt()],index=['总数','最小值','最小值位置','25%分位数',
                    '中位数','75%分位数','均值','最大值','最大值位数','平均绝对偏差','方差','标准差','偏度','峰度'])


def load_raw_data():
    train_data = pd.read_csv('./data/train.csv', low_memory=False, index_col='user_id')
    test_data = pd.read_csv('./data/test.csv', low_memory=False, index_col='user_id')
    return train_data,test_data


def observe_data(train_data,test_data):
    category_features = ["service_type","contract_type","net_service","complaint_level"]

    # contract_time,pay_times,pay_num,
    range_features = ["online_time", "1_total_fee", "2_total_fee","3_total_fee", "4_total_fee", \
                     "last_month_traffic","local_trafffic_month", "local_caller_time", \
                     "service1_caller_time", "service2_caller_time",\
                      "age", "former_complaint_num", "former_complaint_fee"]

    bit_features = ["is_mix_service", "many_over_bill", "is_promise_low_consume", "gender"]
    labels = train_data[['current_service']]
    train_data = train_data.drop(['current_service'], axis=1)
    # get dummpy处理

    # print (test_data['gender'].value_counts())
    #
    # print(test_data['age'].value_counts())
    # print(train_data['age'].value_counts())
    # test_data = pd.get_dummies(test_data, columns=one_hot_feature)

    # train_data['gender'].replace('01', '2', inplace=True)
    # train_data['gender'].replace('02', '2', inplace=True)
    # train_data['gender'].replace('00', '2', inplace=True)
    # train_data['gender'].replace('\\N', '2', inplace=True)
    # train_data['gender'].replace('0', '2', inplace=True)
    # test_data['gender'].replace(0, 2, inplace=True)
    # """ 二值类数据值查看 """
    # for feature in category_features:
    # # for feature in bit_features:
    #     print ("feature: ", feature)
    #     print (train_data[feature].value_counts())
    #     print ("test:  ")
    #     print (test_data[feature].value_counts())
    #     print ("\n")

    # """某一特征项下 不重复数值查看"""
    # for feature in range_features:
    #     print ("feature: ", feature)
    #     print (train_data[feature].value_counts())
    #     print ("\n")

    # train_data.loc[train_data['gender'] == '\\N', 'gender'] = 2
    # train_data.loc[train_data['age'] == '\\N', 'age'] = int(train_data['age'].mode().item())
    # train_data['age'] = pd.to_numeric(train_data['age'])
    # train_data_bin = pd.qcut(train_data['age'], 10, labels = [1,2,3,4,5,6,7,8,9,10])
    # print (train_data_bin[0])

    # QUESRION: could not convert string to float: '\\N' 无法转化？？
    # imp = Imputer(missing_values='\\N', strategy='mean', axis=0)
    # imp.fit(train_data['age'])
    # imp.transform(train_data['age'])

    # """连续数值分箱"""
    # for feature in range_features:
    #     # 所有空值用众数代替
    #     try:
    #         train_data.loc[train_data[feature] == '\\N', feature] = 0
    #         """ 把从文件独处的字符串型数据转化为数值型，一边pd.qcut()进行处理 """
    #         train_data[feature] = pd.to_numeric(train_data[feature])
    #         """ 进行n分位数分箱(从小到大分为n等份) """
    #         train_data_bin = pd.qcut(train_data[feature], 4)
    #         # print (pd.value_counts(train_data_bin))
    #     except:
    #         # 用train_data.loc[train_data[feature] == '\\N', feature] = 0处理报错Invalid Comparision的特征项目
    #         pass
    #
    #     # print (train_data[[feature]].apply(status))
    #     res = train_data[feature].describe()
    #     print (pd.DataFrame(res))

    # """查看与删除重复行"""
    # train_data = train_data.drop_duplicates(keep = "first")
    # print (train_data.duplicated())


    # """ PCA 降维 """
    # print (train_data.columns, len(train_data))
    # pca = PCA(n_components=10, copy=False)
    # pca.fit(train_data)
    # train_data = pca.transform(train_data)
    # print (train_data)

    # print (labels['current_service'].value_counts())
    # mean = train_data['online_time'].mean()
    # mode = train_data['online_time'].mode()
    #
    # print ("net_service", train_data["net_service"].value_counts())
    # print ("complaint_level", train_data["complaint_level"].value_counts())
    # print ("contract_type", train_data["contract_type"].value_counts())

    # dis = Discretization(method='chimerge', max_intervals=20)
    # dis.fit(train_data[['online_time']], None)
    # curr = dis.transform(train_data[['online_time']])

    # print (train_data['1_total_fee'].value_counts(), train_data['online_time'].value_counts())
    # est = KBinsDiscretizer(n_bins=30, encode='ordinal', strategy="kmeans")
    # b_bins = 30
    # for feature in ["1_total_fee", "2_total_fee", "3_total_fee", "4_total_fee"]:
    #     est = KBinsDiscretizer(n_bins=b_bins, encode='ordinal', strategy="kmeans").fit(train_data[feature].values.reshape(-1, 1))
    #     curr = est.transform(train_data[feature].values.reshape(-1, 1))
    #     train_data[feature] = curr
    #     print (train_data[feature].value_counts())
    #     b_bins = b_bins + 5

    est = KBinsDiscretizer(n_bins=45, encode='ordinal', strategy="quantile")
    est.fit(train_data["month_traffic"].values.reshape(-1, 1))
    curr = est.transform(train_data["month_traffic"].values.reshape(-1, 1))
    train_data["month_traffic"] = curr
    print (train_data["month_traffic"].value_counts())

    est = KBinsDiscretizer(n_bins=45, encode='ordinal', strategy="quantile")
    est.fit(train_data["2_total_fee"].values.reshape(-1, 1))
    curr = est.transform(train_data["2_total_fee"].values.reshape(-1, 1))
    train_data["2_total_fee"] = curr
    print(train_data["2_total_fee"].value_counts())
    print("---")


    # for feature in ["1_total_fee", "2_total_fee","3_total_fee", "4_total_fee"]:
    #     train_data[feature] = train_data[feature].replace('\\N', np.nan)
    #     train_data[feature] = train_data[feature].astype(float)
    #     curr_data = train_data.loc[train_data[feature] < 0]
    #     print (curr_data[feature].value_counts())

if __name__ == '__main__':
    train_data,test = load_raw_data()
    observe_data(train_data, test)
