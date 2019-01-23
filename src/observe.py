import numpy as np
import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

fees = ["1_total_fee", "2_total_fee", "3_total_fee", "4_total_fee"]
caller_times = ['local_caller_time', 'service1_caller_time', 'service2_caller_time']
traffic_info = ['month_traffic', 'local_trafffic_month', 'last_month_traffic']
pay_info = ['pay_times', 'pay_num']
others = ['online_time', 'former_complaint_fee','contract_time','former_complaint_num']

range_features = ['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee',
                  'month_traffic','local_trafffic_month', 'last_month_traffic',
                  'local_caller_time', 'service1_caller_time', 'service2_caller_time',
                  'pay_times','pay_num', 'age', 'online_time', 'former_complaint_fee','contract_time']


def load_data():
    train_data = pd.read_csv('./data/train.csv', index_col='user_id')
    test_data = pd.read_csv('./data/test.csv', index_col='user_id')
    train_data = train_data.replace('\\N', np.nan)
    return train_data, test_data


def draw_total_fee(train_data ,test_data):
    X = np.arange(0, len(train_data))
    for feature in fees:
        curr_train_data = train_data[train_data[feature].notna()]
        curr_train_data[feature] = curr_train_data[feature].astype(float)
        print(feature)
        #print (curr_train_data[feature].value_counts())
        observe_data =  curr_train_data.loc[curr_train_data[feature] > 1000, [feature, 'current_service']]
        print (feature, "  %>1000: " + str(len(observe_data)) + " / " +  str(len(train_data)) + ": ", len(observe_data) / len(train_data))
        print (observe_data['current_service'].value_counts())

        # print (observe_data)
        # print ("=======================")
        # observe_data = curr_train_data.loc[ (100 < curr_train_data[feature]) & (1000 > curr_train_data[feature]), [feature, 'current_service']]
        # print (observe_data['current_service'].value_counts())

    total_fee_1 = curr_train_data['1_total_fee'].values
    total_fee_2 = curr_train_data['2_total_fee'].values
    total_fee_3 = curr_train_data['3_total_fee'].values
    total_fee_4 = curr_train_data['4_total_fee'].values

    fig = plt.figure()
    plt.title('1_total_fee & 2_total_fee & 3_total_fee & 4_total_fee')
    # 注意add_subplot()的使用以及add_subplot()参数含义同plt.subplot()
    ax1 = fig.add_subplot(221)
    ax1.scatter(X,total_fee_1, s = 30, c = 'red', marker = 's')
    ax2 = fig.add_subplot(222)
    ax2.scatter(X,total_fee_2, s = 30, c = 'green', marker = 's')
    ax3 = fig.add_subplot(223)
    ax3.scatter(X,total_fee_3, s = 30, c = 'brown', marker = 's')
    ax4 = fig.add_subplot(224)
    ax4.scatter(X,total_fee_4, s = 30, c = 'blue', marker = 's')
    plt.show()

    # print (train_data[(train_data['1_total_fee'] > 4000)])
    # print (train_data[ (train_data['1_total_fee'] > 4000) or (train_data['2_total_fee'] > 4000) or (train_data['3_total_fee'] > 4000) or (train_data['4_total_fee'] > 4000) ])

def draw_traffic_info(train_data,test_data):

    X = np.arange(0, len(train_data))
    for feature in caller_times:
        curr_train_data = train_data[train_data[feature].notna()]
        curr_train_data[feature] = curr_train_data[feature].astype(float)
        # observe_data = curr_train_data.loc[curr_train_data[feature] < 1000, [feature, 'current_service']]
        # print ("=======================")
        # observe_data = curr_train_data.loc[ (100 < curr_train_data[feature]) & (1000 > curr_train_data[feature]), [feature, 'current_service']]
        # print(observe_data['current_service'].value_counts())

    feature = 'month_traffic'
    observe_data = curr_train_data.loc[curr_train_data[feature] > 40000, [feature, 'current_service']]
    print(feature, "  %>40000: " + str(len(observe_data)) + " / " + str(len(train_data)) + ": ",
          len(observe_data) / len(train_data))
    print(observe_data['current_service'].value_counts())

    feature = 'local_trafffic_month'
    observe_data = curr_train_data.loc[curr_train_data[feature] > 25000, [feature, 'current_service']]
    print(feature, "  %>25000: " + str(len(observe_data)) + " / " + str(len(train_data)) + ": ",
          len(observe_data) / len(train_data))
    print(observe_data['current_service'].value_counts())


    month_traffic = curr_train_data['month_traffic'].values
    local_trafffic_month = curr_train_data['local_trafffic_month'].values
    last_month_traffic = curr_train_data['last_month_traffic'].values


    fig = plt.figure()
    plt.title('month_traffic & local_trafffic_month & last_month_traffic')
    # 注意add_subplot()的使用以及add_subplot()参数含义同plt.subplot()
    ax1 = fig.add_subplot(131)
    ax1.scatter(X, month_traffic, s=30, c='red', marker='s')
    ax2 = fig.add_subplot(132)
    ax2.scatter(X, local_trafffic_month, s=30, c='green', marker='s')
    ax3 = fig.add_subplot(133)
    ax3.scatter(X, last_month_traffic, s=30, c='blue', marker='s')
    plt.show()


def draw_caller_time(train_data,test_data):

    X = np.arange(0, len(train_data))
    for feature in caller_times:
        curr_train_data = train_data[train_data[feature].notna()]
        curr_train_data[feature] = curr_train_data[feature].astype(float)
        observe_data = curr_train_data.loc[curr_train_data[feature] > 1000, [feature, 'current_service']]
        print(feature, "  %>1000: " + str(len(observe_data)) + " / " + str(len(train_data)) + ": ", len(observe_data) / len(train_data))
        print(observe_data['current_service'].value_counts())
        # print ("=======================")
        # observe_data = curr_train_data.loc[ (100 < curr_train_data[feature]) & (1000 > curr_train_data[feature]), [feature, 'current_service']]
        # print(observe_data['current_service'].value_counts())

    local_caller_time = curr_train_data['local_caller_time'].values
    service1_caller_time = curr_train_data['service1_caller_time'].values
    service2_caller_time = curr_train_data['service2_caller_time'].values

    fig = plt.figure()
    plt.title('local_caller_time & service1_caller_time & service2_caller_time')
    # 注意add_subplot()的使用以及add_subplot()参数含义同plt.subplot()
    ax1 = fig.add_subplot(131)
    ax1.scatter(X, local_caller_time, s=30, c='red', marker='s')
    ax2 = fig.add_subplot(132)
    ax2.scatter(X, service1_caller_time, s=30, c='green', marker='s')
    ax3 = fig.add_subplot(133)
    ax3.scatter(X, service1_caller_time, s=30, c='brown', marker='s')
    plt.show()


def draw_age_info(train_data,test_data):

    curr_train_data = train_data[train_data['age'].notna()]
    curr_train_data['age'] = curr_train_data['age'].astype(float)
    X = np.arange(0, len(curr_train_data))
    observe_data = curr_train_data.loc[curr_train_data['age'] > 60, ['age', 'current_service']]
    print('age', "  %>55: " + str(len(observe_data)) + " / " + str(len(train_data)) + ": ", len(observe_data) / len(train_data))
    print(observe_data['current_service'].value_counts())
    observe_data = curr_train_data.loc[curr_train_data['age'] < 20, ['age', 'current_service']]
    print('age', "  %<25: " + str(len(observe_data)) + " / " + str(len(train_data)) + ": ",
          len(observe_data) / len(train_data))
    print(observe_data['current_service'].value_counts())
    # print ("=======================")
    # observe_data = curr_train_data.loc[ (100 < curr_train_data[feature]) & (1000 > curr_train_data[feature]), [feature, 'current_service']]
    # print(observe_data['current_service'].value_counts())

    ages = curr_train_data['age'].values

    # fig = plt.figure()
    # plt.title('age')
    # # 注意add_subplot()的使用以及add_subplot()参数含义同plt.subplot()
    # ax1 = fig.add_subplot(111)
    # ax1.scatter(X, ages, s=30, c='red', marker='s')
    # plt.show()

def draw_pay_info(train_data, test_data):
    X = np.arange(0, len(train_data))
    for feature in pay_info:
        curr_train_data = train_data[train_data[feature].notna()]
        curr_train_data[feature] = curr_train_data[feature].astype(float)

    feature = 'pay_num'
    print (feature)
    observe_data = curr_train_data.loc[curr_train_data[feature] > 500, [feature, 'current_service']]
    print(feature, "  %>500: " + str(len(observe_data)) + " / " + str(len(train_data)) + ": ",len(observe_data) / len(train_data))
    print(observe_data['current_service'].value_counts())

    feature = 'pay_times'
    print (feature)
    observe_data = curr_train_data.loc[curr_train_data[feature] > 12, [feature, 'current_service']]
    print(feature, "  %>12: " + str(len(observe_data)) + " / " + str(len(train_data)) + ": ",
          len(observe_data) / len(train_data))
    print(observe_data['current_service'].value_counts())

    pay_num = curr_train_data['pay_num'].values
    pay_times = curr_train_data['pay_times'].values

    fig = plt.figure()
    plt.title('Pay num & Pay times')
    # 注意add_subplot()的使用以及add_subplot()参数含义同plt.subplot()
    ax1 = fig.add_subplot(121)
    ax1.scatter(X, pay_num, s=30, c='red', marker='s')
    ax2 = fig.add_subplot(122)
    ax2.scatter(X, pay_times, s=30, c='green', marker='s')
    plt.show()

def draw_other_info(train_data, test_data):
    X = np.arange(0, len(train_data))
    for feature in pay_info:
        curr_train_data = train_data[train_data[feature].notna()]
        curr_train_data[feature] = curr_train_data[feature].astype(float)

    # feature = 'former_complaint_fee'
    # print (feature)
    # observe_data = curr_train_data.loc[curr_train_data[feature] > 0.15, [feature, 'current_service']]
    # print(feature, "  %>0.15: " + str(len(observe_data)) + " / " + str(len(train_data)) + ": ",len(observe_data) / len(train_data))
    # print(observe_data['current_service'].value_counts())

    # feature = 'contract_time'
    # print (feature)
    # observe_data = curr_train_data.loc[curr_train_data[feature] > 35, [feature, 'current_service']]
    # print(feature, "  %>35: " + str(len(observe_data)) + " / " + str(len(train_data)) + ": ",
    #       len(observe_data) / len(train_data))
    # print(observe_data['current_service'].value_counts())
    #
    # observe_data = curr_train_data.loc[(curr_train_data[feature] == 0), [feature, 'current_service']]
    # print(feature, "  % =0: " + str(len(observe_data)) + " / " + str(len(train_data)) + ": ",
    #       len(observe_data) / len(train_data))
    # print(observe_data['current_service'].value_counts())

    feature = 'former_complaint_num'
    print (feature)
    observe_data = curr_train_data.loc[curr_train_data[feature] > 5, [feature, 'current_service']]
    print(feature, "  %>5: " + str(len(observe_data)) + " / " + str(len(train_data)) + ": ",len(observe_data) / len(train_data))
    print(observe_data['current_service'].value_counts())


    online_time = curr_train_data['online_time'].values
    former_complaint_fee = curr_train_data['former_complaint_fee'].values
    contract_time = curr_train_data['contract_time'].values
    former_complaint_num = curr_train_data['former_complaint_num'].values


    fig = plt.figure()
    plt.title('online_time, former_complaint_fee, contract_time')
    # 注意add_subplot()的使用以及add_subplot()参数含义同plt.subplot()
    ax1 = fig.add_subplot(221)
    ax1.scatter(X, online_time, s=30, c='red', marker='s')
    ax2 = fig.add_subplot(222)
    ax2.scatter(X, former_complaint_fee, s=30, c='green', marker='s')
    ax3 = fig.add_subplot(223)
    ax3.scatter(X, contract_time, s=30, c='green', marker='s')
    ax4 = fig.add_subplot(224)
    ax4.scatter(X, former_complaint_num, s=30, c='green', marker='s')
    plt.show()



def draw_month_info(train_data, test_data):
    month_info = ["local_trafffic_month", "local_caller_time", "1_total_fee"]
    for feature in month_info:
        curr_train_data = train_data[train_data[feature].notna()]
        curr_train_data[feature] = curr_train_data[feature].astype(float)
        # observe_data = curr_train_data.loc[curr_train_data[feature] < 1000, [feature, 'current_service']]
        # print ("=======================")
        # observe_data = curr_train_data.loc[ (100 < curr_train_data[feature]) & (1000 > curr_train_data[feature]), [feature, 'current_service']]
        # print(observe_data['current_service'].value_counts())


if __name__ == '__main__':
    train_data, test_data = load_data()
    draw_total_fee(train_data, test_data)
    # draw_caller_time(train_data, test_data)
    # draw_traffic_info(train_data, test_data)
    # draw_pay_info(train_data, test_data)
    # draw_other_info(train_data, test_data)
    # draw_age_info(train_data, test_data)




