# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from random import choice
# from utils import middle
import warnings
warnings.filterwarnings('ignore')

def load_data():
    print('Loading dataset...')
    test_data = pd.read_csv('./data/test.csv', low_memory=False, index_col='user_id')
    train_data = pd.read_csv('./data/train.csv', low_memory=False, index_col='user_id')
    print('Loading over ... ... ')

    one_hot_feature = ["is_mix_service", "many_over_bill", "is_promise_low_consume", \
                       "gender", "service_type", "contract_type", "net_service", "complaint_level"]

    range_features = ['online_time', '1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee', 'month_traffic',
                      'last_month_traffic', 'local_trafffic_month', 'local_caller_time', 'service1_caller_time',
                      'service2_caller_time', 'age', 'former_complaint_fee', 'contract_time', 'pay_times', 'pay_num',
                      'former_complaint_num']
    # 89016252，89016259 ，89016253 ，99104722
    train_data = train_data[train_data['current_service'] != 89016252]
    train_data = train_data[train_data['current_service'] != 89016259]
    train_data = train_data[train_data['current_service'] != 89016253]
    train_data = train_data[train_data['current_service'] != 99104722]

    train_data = train_data.drop(['service_type'], axis=1)
    test_data = test_data.drop(['service_type'], axis=1)
    one_hot_feature.remove('service_type')

    train_data = train_data.drop(['net_service'], axis=1)
    test_data = test_data.drop(['net_service'], axis=1)
    one_hot_feature.remove('net_service')

    train_data = train_data.drop(["complaint_level"], axis=1)
    test_data = test_data.drop(["complaint_level"], axis=1)
    one_hot_feature.remove("complaint_level")

    labels = train_data[['current_service']]
    train_data = train_data.drop(['current_service'], axis=1)

    # train_data = train_data.replace('\\N', np.nan)
    # test_data = test_data.replace('\\N', np.nan)

    genders = [0, 1]
    try:
        # 性别异常和缺失值处理

        train_data.loc[train_data['gender'] == '01', 'gender'] = 0
        train_data.loc[train_data['gender'] == '02', 'gender'] = 1
        train_data.loc[train_data['gender'] == '00', 'gender'] = choice(genders)
        train_data.loc[train_data['gender'] ==  '\\N', 'gender'] = choice(genders)
        train_data.loc[train_data['gender'] == '2', 'gender'] = choice(genders)
        train_data['gender'] = train_data['gender'].astype(str)

        test_data['gender'] = test_data['gender'].replace('\\N', np.nan)
        test_data.loc[test_data['gender'] == np.nan, 'gender'] = choice(genders)
        test_data.loc[test_data['gender'] == 2, 'gender'] = choice(genders)
        test_data['gender'] = test_data['gender'].astype(str)
    except:
        pass

    # complaint_levels = [1, 2, 3]
    # try:
    #     train_data.loc[train_data['complaint_level'] == '0', 'complaint_level'] = choice(complaint_levels)
    #     test_data.loc[test_data['complaint_level'] == '0', 'complaint_level'] = choice(complaint_levels)
    # except:
    #     pass

    limit_map = {'1_total_fee': 1000, '2_total_fee': 1000, '3_total_fee': 1000, '4_total_fee': 1000,
                 'local_caller_time': 1000, 'service1_caller_time': 1000, 'service2_caller_time': 1000,
                 'month_traffic': 40000, 'local_trafffic_month': 25000, 'pay_num': 500, 'pay_times': 12,
                 'former_complaint_fee': 0.15, 'contract_time': 35, 'former_complaint_num': 5}

    # 所有数值特征的异常值置为0
    # ss = StandardScaler()
    for feature in range_features:

        train_data[feature] = train_data[feature].replace('\\N', np.nan)
        train_data[feature] = train_data[feature].astype(float)
        # train_data.loc[train_data[feature] < 0, feature] = np.nan

        test_data[feature] = test_data[feature].replace('\\N', np.nan)
        test_data[feature] = test_data[feature].astype(float)
        # test_data.loc[test_data[feature] < 0, feature] = np.nan

        try:
            # item = float(train_data[feature].mean().item())
            # train_data[feature] = train_data[feature].fillna(item).astype(float)
            if feature in limit_map:
                train_data.loc[train_data[feature] > limit_map[feature], feature] = float(limit_map[feature])
            # if feature == 'age':
            #     train_data.loc[train_data[feature] < 25, feature] = item
        except:
            pass

        try:
            # item = float(test_data[feature].mean().item())
            # test_data[feature] = test_data[feature].fillna(item).astype(float)
            if feature in limit_map:
                test_data.loc[test_data[feature] > limit_map[feature], feature] = float(limit_map[feature])
            # if feature == 'age':
            #     test_data.loc[test_data[feature] < 25, feature] = item
        except:
            pass

    train = train_data[range_features].astype('float64')
    test = test_data[range_features].astype('float64')
    
    # 特征运算
    # train['total_fee'] = train['1_total_fee'] + train['2_total_fee'] + train['3_total_fee'] + train['4_total_fee']
    # test['total_fee'] = test['1_total_fee'] + test['2_total_fee'] + test['3_total_fee'] + test['4_total_fee']
    # #
    # train['total_fee/online_time'] = train['total_fee'] / train['online_time']
    # test['total_fee/online_time'] = test['total_fee'] / test['online_time']
    #
    # train['month_traffic/online_time'] = train['month_traffic'] / train['online_time']
    # test['month_traffic/online_time'] = test['month_traffic'] / test['online_time']
    #
    # train['1_total_fee/month_traffic'] = train['1_total_fee'] / train['month_traffic']
    # test['1_total_fee/month_traffic'] = test['1_total_fee'] / test['month_traffic']
    #
    # train['caller_time'] = train['local_caller_time'] + train['service1_caller_time'] + train['service2_caller_time']
    # test['caller_time'] = test['local_caller_time'] + test['service1_caller_time'] + test['service2_caller_time']
    # #
    # train['total_fee/caller_time'] = train['total_fee'] / train['caller_time']
    # test['total_fee/caller_time'] = test['total_fee'] / test['caller_time']

    # 特征分箱
    # est = KBinsDiscretizer(n_bins=30, encode='ordinal', strategy="kmeans")
    # est.fit(train['online_time'].values.reshape(-1, 1))
    # train['online_time_new'] = est.transform(train['online_time'].values.reshape(-1, 1))
    # test['online_time_new'] = est.transform(test['online_time'].values.reshape(-1, 1))
    # """为什么处理 2/3/4_total_fee报错？"""
    # est = KBinsDiscretizer(n_bins=40, encode='ordinal', strategy="kmeans")
    # est.fit(train['1_total_fee'].values.reshape(-1, 1))
    # train['1_total_fee_new'] = est.transform(train['1_total_fee'].values.reshape(-1, 1))
    # test['1_total_fee_new'] = est.transform(test['1_total_fee'].values.reshape(-1, 1))
    # #print(train_data["1_total_fee"].value_counts())
    #
    # est = KBinsDiscretizer(n_bins=40, encode='ordinal', strategy="kmeans")
    # est.fit(train["service2_caller_time"].values.reshape(-1, 1))
    # train["service2_caller_time_new"] = est.transform(train["service2_caller_time"].values.reshape(-1, 1))
    # test["service2_caller_time_new"] = est.transform(test["service2_caller_time"].values.reshape(-1, 1))
    #print (train_data["service2_caller_time"].value_counts())

    # est = KBinsDiscretizer(n_bins=45, encode='ordinal', strategy="quantile")
    # est.fit(train["2_total_fee"].values.reshape(-1, 1))
    # train["2_total_fee_new"] = est.transform(train["2_total_fee"].values.reshape(-1, 1))
    # test["2_total_fee_new"] = est.transform(test["2_total_fee"].values.reshape(-1, 1))

    # est = KBinsDiscretizer(n_bins=45, encode='ordinal', strategy="quantile")
    # est.fit(train["month_traffic"].values.reshape(-1, 1))
    # train["month_traffic_new"] = est.transform(train["month_traffic"].values.reshape(-1, 1))
    # test["month_traffic_new"] = est.transform(test["month_traffic"].values.reshape(-1, 1))
    
    # One-hot编码
    enc = OneHotEncoder()
    for feature in one_hot_feature:

        curr_feature_data = train_data[train_data[feature].notna()]
        enc.fit(train_data[feature].values.reshape(-1, 1))
        train_a = enc.transform(train_data[feature].values.reshape(-1, 1))
        train_a = train_a.toarray()
        test_a = enc.transform(test_data[feature].values.reshape(-1, 1))
        test_a = test_a.toarray()
        train_b, test_b = pd.DataFrame(train_a, index=train.index), pd.DataFrame(test_a, index=test.index)
        train_b.columns, test_b.columns = [list(map(lambda x: feature + '_' + str(x), range(train_a.shape[1])))] * 2
        train, test = pd.concat([train, train_b], axis=1), pd.concat([test, test_b], axis=1)

    return train, labels, test

if __name__ == '__main__':
    load_data()