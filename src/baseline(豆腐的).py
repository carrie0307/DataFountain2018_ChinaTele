'''
最近正在筹备一个网站，初步计划是分享一些资料，后续发展再议。 网站：http://39.107.234.237/
'''

# 导入数据包
import pandas as pd
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# 基础配置信息
path = '../data/'
n_splits = 5
seed = 42

# lgb 参数
params={
    "learning_rate":0.1,
    "lambda_l1":0.1,
    "lambda_l2":0.2,
    "max_depth":4,
    "objective":"multiclass",
    "num_class":15,
    "silent":True,
}

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

# 构造原始数据
y = train.pop('current_service')
train_id = train.pop('user_id')
# 这个字段有点问题
X = train
train_col = train.columns
X_test = test[train_col]
test_id = test['user_id']

# 数据有问题数据
for i in train_col:
    X[i] = X[i].replace("\\N",-1)
    X_test[i] = X_test[i].replace("\\N",-1)

X,y,X_test = X.values,y,X_test.values

# 采取k折模型方案
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np

# 自定义F1评价函数
def f1_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    preds = np.argmax(preds.reshape(15, -1), axis=0)
    score_vali = f1_score(y_true=labels, y_pred=preds, average='weighted')
    return 'f1_score', score_vali, True

xx_score = []
cv_pred = []

skf = StratifiedKFold(n_splits=n_splits,random_state=seed,shuffle=True)
for index,(train_index,test_index) in enumerate(skf.split(X,y)):
    print(index)

    X_train,X_valid,y_train,y_valid = X[train_index],X[test_index],y[train_index],y[test_index]

    train_data = lgb.Dataset(X_train, label=y_train)
    validation_data = lgb.Dataset(X_valid, label=y_valid)

    clf=lgb.train(params,train_data,num_boost_round=100000,valid_sets=[validation_data],early_stopping_rounds=50,feval=f1_score_vali,verbose_eval=1)

    xx_pred = clf.predict(X_valid)

    xx_pred = [np.argmax(x) for x in xx_pred]

    xx_score.append(f1_score(y_valid,xx_pred,average='weighted'))

    y_test = clf.predict(X_test)

    y_test = [np.argmax(x) for x in y_test]

    if index == 0:
        cv_pred = np.array(y_test).reshape(-1, 1)
    else:
        cv_pred = np.hstack((cv_pred, np.array(y_test).reshape(-1, 1)))

# 投票
submit = []
for line in cv_pred:
    submit.append(np.argmax(np.bincount(line)))

# 保存结果
df_test = pd.DataFrame()
df_test['id'] = list(test_id.unique())
df_test['predict'] = submit
df_test['predict'] = df_test['predict'].map(label2current_service)

df_test.to_csv('../result/baseline2.csv',index=False)

print(xx_score,np.mean(xx_score))