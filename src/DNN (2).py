from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation, BatchNormalization
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import Callback
#from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
import keras.backend as K
from sklearn.preprocessing import OneHotEncoder

def f1_score(y_true, y_pred):
    print(y_true)
    print(y_pred)
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


path = '../input/'
# 读取数据
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')

'''
简单分析数据：
user_id 为编码后的数据，大小：
train data shape (612652, 27)
train data of user_id shape 612652
简单的1个用户1条样本的题目,标签的范围 current_service
'''
print('标签',set(train.columns)-set(test.columns))

print('train data shape',train.shape)
print('train data of user_id shape',len(set(train['user_id'])))
print('train data of current_service shape',(set(train['current_service'])))

print('train data shape',test.shape)
print('train data of user_id shape',len(set(test['user_id'])))

# 对标签编码 映射关系
label2current_service = dict(zip(range(0,len(set(train['current_service']))),sorted(list(set(train['current_service'])))))
current_service2label = dict(zip(sorted(list(set(train['current_service']))),range(0,len(set(train['current_service'])))))

# 原始数据的标签映射
train['current_service'] = train['current_service'].map(current_service2label)

# 这个字段有点问题
X = train
train_col = test.columns

# X_test = test[train_col]

# 数据有问题数据
for i in train_col:
    train[i] = train[i].replace("\\N",-1)
    test[i] = test[i].replace("\\N",-1)

train_id = train.pop('user_id')
test_id = test.pop('user_id')

columns=test.columns
##需要独热编码的特征
one_hot=OneHotEncoder()
la=LabelEncoder()
feat_one_hot=['service_type','is_mix_service','many_over_bill','contract_type','is_promise_low_consume','net_service','gender']
for feat in feat_one_hot:
    train[feat]=train[feat].map(lambda x: int(x))
    test[feat]=test[feat].map(lambda x: int(x))
    train_tem=pd.get_dummies(train[feat])
    train_tem.columns=[feat+str(i) for i in range(len(train_tem.columns))]
    train=pd.concat([train,train_tem],axis=1)
    test_tem=pd.get_dummies(test[feat])
    test_tem.columns=[feat+str(i) for i in range(len(test_tem.columns))]
    test=pd.concat([test,test_tem],axis=1)
    del(train[feat])
    del(test[feat])
feat_normal=[item for item in columns if item not in feat_one_hot]

sc=StandardScaler()
test_=test
test_[feat_normal]=sc.fit_transform(test[feat_normal])
la.fit(train['current_service'])
target=pd.get_dummies(train['current_service'])
del(train['current_service'])
train_=train
train_[feat_normal]=sc.fit_transform(train[feat_normal])
#x_train,x_test,y_train,y_test=train_test_split(train_,target,random_state=1)

# 自定义F1评价函数
def f1_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    preds = np.argmax(preds.reshape(15, -1), axis=0)
    score_vali = f1_score(y_true=labels, y_pred=preds, average='marco')
    return 'f1_score', score_vali, True



dnn_Model=Sequential()
dnn_Model.add(Dense(512,input_dim=train_.shape[1],activation='relu'))
dnn_Model.add(BatchNormalization())
dnn_Model.add(Dropout(0.5))
dnn_Model.add(Dense(1024,activation='relu'))
dnn_Model.add(BatchNormalization())
dnn_Model.add(Dropout(0.5))
dnn_Model.add(Dense(1024,activation='relu'))
dnn_Model.add(BatchNormalization())
dnn_Model.add(Dropout(0.5))
dnn_Model.add(Dense(1024,activation='relu'))
dnn_Model.add(BatchNormalization())
dnn_Model.add(Dropout(0.5))
dnn_Model.add(Dense(1024,activation='relu'))
dnn_Model.add(BatchNormalization())
dnn_Model.add(Dropout(0.5))
dnn_Model.add(Dense(1024,activation='relu'))
dnn_Model.add(BatchNormalization())
dnn_Model.add(Dropout(0.2))
dnn_Model.add(Dense(15,activation='softmax'))

print([i for i in train_.columns if i not in test_.columns])


dnn_Model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy',f1_score])

dnn_Model.fit(train_,target,batch_size=512,epochs=20,callbacks=[ReduceLROnPlateau(factor=0.1, patience=2, min_lr=0.000001)])



preds=dnn_Model.predict(test_)
preds=[np.argmax(i) for i in preds]
preds=la.inverse_transform(preds)

df_test = pd.DataFrame()
df_test['id'] = list(test_id.unique())
df_test['predict'] = preds
df_test['predict'] = df_test['predict'].map(label2current_service)

print(df_test['predict'].value_counts())
df_test.to_csv('../sub/dnnbaseline2.csv',index=False)

