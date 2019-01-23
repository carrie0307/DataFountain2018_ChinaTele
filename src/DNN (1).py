from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import Callback
#from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.callbacks import EarlyStopping
import keras.backend as K
from sklearn.preprocessing import OneHotEncoder

def f1_score(y_true, y_pred):

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

early_stoping=EarlyStopping(patience=10)


train=pd.read_csv('train_cleaned.csv')
train.sample(frac=1).reset_index(drop=True)
test=pd.read_csv('test_cleaned.csv')
results=pd.read_csv('submit_sample.csv')

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
    train_tem.columns=[feat+str(i) for i in range(len(train_tem.columns))]
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


dnn_Model=Sequential()
dnn_Model.add(Dense(512,input_dim=50,activation='relu'))
dnn_Model.add(Dropout(0.5))
dnn_Model.add(Dense(1024,activation='relu'))
dnn_Model.add(Dropout(0.5))
dnn_Model.add(Dense(1024,activation='relu'))
dnn_Model.add(Dropout(0.5))
dnn_Model.add(Dense(1024,activation='relu'))
dnn_Model.add(Dropout(0.5))
dnn_Model.add(Dense(1024,activation='relu'))
dnn_Model.add(Dropout(0.5))
dnn_Model.add(Dense(1024,activation='relu'))
dnn_Model.add(Dropout(0.5))
dnn_Model.add(Dense(15,activation='softmax'))


dnn_Model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy',f1_score])

dnn_Model.fit(train_,target,batch_size=10000,epochs=80,callbacks=[early_stoping],validation_split=0.2)



preds=dnn_Model.predict(test_)
preds=[np.argmax(i) for i in preds]
preds=la.inverse_transform(preds)
results['predict']=preds
results.to_csv('dnn_results.csv',index=0)
