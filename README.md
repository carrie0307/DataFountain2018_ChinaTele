# 面向电信行业存量用户的智能套餐个性化匹配模型-个人记录

* 问题类型: 分类

* 作者
    
    * [@lsvih](https://github.com/lsvih)

    * [@carrie0307](https://github.com/carrie0307)

## 最最最最最基本baseline

根据大佬指导接把特征放到baseline就能跑到63%,相关baseline记录如下

### 数据读取与初步处理部分

* 数据读取: pd.read_csv()

* 数据空值处理：此题中空值不是np.nan,而是'\\N'。因此在处理前需要先观察数据，暂且将无法处理的'\\N'处理成np.nan,使数据能跑起来。

* 特征分类处理：大体分为两类，分别进行最简单的预处理

    * **数值型特征**：可以先简单把空值处理成np.nan;这次比赛中有些数值特征读出来的str型号，因此建议统一转为float等数值类型。

    * **类别型特征**:进行最基本的one_hot编码，通过sklearn.preprocessing.OneHotEncoder进行即可，但要注意的是：
    OneHotEncoder().fit_transform()得到的是一个**dataframe**，不能直接作为原dataframe中的一个seires，因此要进行转换。大致代码如下：

    ```python
    enc = OneHotEncoder()

    train_a = enc.transform(train_data[feature].values.reshape(-1, 1)).toarray()
    test_a = enc.transform(test_data[feature].values.reshape(-1, 1)).toarray()
    train_b, test_b = pd.DataFrame(train_a, index=train.index), pd.DataFrame(test_a, index=test.index)
    # 注意这一步
    train_b.columns, test_b.columns = [list(map(lambda x: feature + '_' + str(x), range(train_a.shape[1])))] * 2
    # train是之前值包括数值型特征的dataframe，通过axis=1(行对齐)进行合并
    train, test = pd.concat([train, train_b], axis=1), pd.concat([test, test_b], axis=1)
    ```

* 训练集与测试集

```python
    
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
```

### 模型

* 以上数据处理好后，直接放模型跑即可。这次初步尝试的是决策树和随机森林，很基本的代码sklearn文档都有，这里贴出来一个例子

```python

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score

def randomforest_Classifier():

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

```

后续使用了XGB和LGB,这些代码部分较之一般模型稍有不同，后续详细列出。

## 数值特征调整

### 数据调整-1：数据观察与异常值处理

当以上基本数据放入XGB也跑不出新的提高时，开始了数据处理，主要是针对数值型特征。进行过的处理如下

* 首先通过pandas.describe(df[feature])查看数据分布(但自己好像没看出什么来)，然后通过**画图查看了数据分布**，这一点觉得十分有效，能够明显地观察到数据的分布尤其是一些异常离群数据(例如，明显拖在尾巴上的数据)。就是在进行这一步的过程中，辅助图对离群的点进行重新赋值,例如下图关于四个total_fee的分布查看

![](http://ouzh4pejg.bkt.clouddn.com/togal%20fee%20observe.png)

值得注意的是，将(可能)具有关系的数值画在同一张图上，便于进行观察。基于对图中数据分布的观察，在代码中写了limit_map,对异常值的点进行了处理，而结果确实得到了提高。

### 数据调整-2:空值处理

对于数值型特征，除了**数据调整-1**的部分，还将np.nan(即原先的\\n)赋值为了**众数**或**均值**(但似乎没什么效果。。。。)

### 数据调整-3：特征运算

在群里看到说可以对特征进行一定的运算，最基本就是**加减乘除**,在这个题目中主要根据一些(可能)存在关联的数据进行了除法，例如四种total_fee之和与在线时间、流量使用总量的比值等。实际上我进行了这一操作并没有带来提高，但**挺多资料都推荐了对特征的运算，因此记录下这一思想**。

### 数据调整-4：特征评估

* 首先通过决策树查看了特征权重，然后又通过PCA进行了查看，最后得到的结果大致是一致的，相关代码如下


```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

feature_names = X_train.columns

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

```

### 数据调整-7：标准化 or 归一化

* 用preprocessing.StandardScaler().fit(X)进行了处理，但是没有得到效果的提高。

* 一篇文章：https://www.cnblogs.com/chaosimple/p/4153167.html

### 数据调整-6：分箱

其实很早就想做分箱了，一般情况下通过pandas.cut()或pandas.qcut()即可进行，但一直困惑于**应该分为几箱和如何确定分箱间隔**,因此一直没有做。后来发现了sklearn.preprocessing.KBinsDiscretizer(scikit-learn v0.20.0)，可以设定**分箱数**和**计算算法**自动进行分箱，为了省事用了这个。关于**分箱数**如何确定仍旧没有明确，只是自己在结合pandas.describe()的情况在进行试验，但通过对service2_caller_time和online_time(特征评估中权重最高的两个特征)进行分箱后的确有了提高(但后续再添加分箱后遍没有了。。。)


## 模型

### XGB

* XGB Documentation: https://xgboost.readthedocs.io/en/latest/python/python_api.html

* 据说Kaggle前几名都用XGB，跑了后的确发现效果提升明显，现贴出主要代码及相关注释

* XGB支持GPU，如果需要的话，在参数中加上**'tree_method':'gpu_hist'**,如果不需要指定CPU,直接运行即可。

* XGB运行很慢，建议使用GPU。

```python

"""
代码是从原先7折代码里摘出来的，摘出后没有运行，但应该没有错误
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cross_validation import train_test_split
import xgboost as xgb

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

def f1_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    preds = np.argmax(preds.reshape(15, -1), axis=0)
    score_vali = f1_score(y_true=labels, y_pred=preds, average='weighted')
    return 'f1_score', score_vali, True

def train_model(X,y,test)
        

        X_train,X_valid,y_train,y_valid = train_test_split(x,y,test_size=0.3,random_state=0)

        # 注意XGB的数据转换
        X_test = xgb.DMatrix(data=test)
        train_data = xgb.DMatrix(data=X_train, label=y_train)
        validation_data = xgb.DMatrix(data=X_valid, label=y_valid)

        print ("Training ....")
        clf = xgb.train(params, train_data, num_boost_round=2000,
                        evals=[(train_data, 'train'), (validation_data, 'eval')], early_stopping_rounds=50, verbose_eval=1)
        print ("Save model ...")
        clf.save_model(path + '/models/xgb')

        print ("Process X_valid ...")
        xx_pred = clf.predict(validation_data)
        xx_pred = np.argmax(xx_pred, axis=1)
        xx_score = f1_score(y_valid, xx_pred, average='weighted')

        print ("Process X_test ...")
        y_test = clf.predict(X_test)
        submit = np.argmax(y_test, axis=1)
        return submit

# 注意：假设要让XGB分n类，则输入的标签应当是0~n-1,否则会报错，课通过如下代码转换。
service_label = [89950166, 89950167, 99999828, 89950168, 99999827, 99999826,\
                     99999830, 99999825, 90063345, 90109916,90155946]
label_set = {k: i for i, k in enumerate(service_label)}
for service in label_set.keys():
    labels.loc[labels['current_service'] == service, 'current_service'] = label_set[service]

# 进行训练
submit = train_model(train, labels, test)

# 再对训练出的label进行转换:从0~n-1转为实际的标签
for i, v in enumerate(submit):
    submit[i] = service_label[v]

```

### LGB

* LGB Documentation
    * https://lightgbm.readthedocs.io/en/latest/Python-Intro.html

    * https://lightgbm.readthedocs.io/en/latest/Python-API.html

* LGB是轻量的XGB，代码如下

```python
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cross_validation import train_test_split
import lightgbm as lgb

# LGB参数
params = {'learning_rate': 0.05, 'max_depth': 6, 'min_child_weight': 1, 'seed': 0,
          'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0, 'reg_lambda': 1, 'n_jobs': -1,
          'num_class': 11, 'objective': 'multiclass', "lambda_l1": 0.1, "lambda_l2": 0.2}

def f1_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    preds = np.argmax(preds.reshape(11, -1), axis=0)
    score_vali = f1_score(y_true=labels, y_pred=preds, average='weighted')
    return 'f1_score', score_vali, True

#  X_train,X_valid,y_train,y_valid = train_test_split(x,y,test_size=0.3,random_state=0)
def train_model(X, y, test):
    
    # 注意test不用进行转换；
    X_test = test
    X_train,X_valid,y_train,y_valid = train_test_split(x,y,test_size=0.3,random_state=0) 
    # 注意：X_valid不用进行转换
    train_data = lgb.Dataset(data=X_train, label=y_train)

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
    xx_score = f1_score(y_valid, xx_pred, average='weighted')

    print ("Process X_test ...")
    y_test = clf.predict(X_test)
    submit = np.argmax(y_test, axis=1)
    return submit

train, labels, test = load_data()
# LGB同样需要对标签进行转换
service_label = [89950166, 89950167, 99999828, 89950168, 99999827, 99999826,\
                 99999830, 99999825, 90063345, 90109916,90155946]
label_set = {k: i for i, k in enumerate(service_label)}
for service in label_set.keys():
    labels.loc[labels['current_service'] == service, 'current_service'] = label_set[service]
# 训练
submit = train_model(train, labels, test)
# 标签还原
for i, v in enumerate(submit):
    submit[i] = service_label[v]

```


## 训练

### k-fold & 结果投票

* 注意：np.argmax(np.bincount(line))

```python

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from new_dataset import load_data

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
        clf = lgb.train(params, train_data, num_boost_round=3000, valid_sets=[validation_data],early_stopping_rounds=50, feval=f1_score_vali, verbose_eval=1)
        print ("Save model ...")
        clf.save_model(path + '/models/lgb_' + str(index))

        print ("Process X_valid ...")
        xx_pred = clf.predict(X_valid)
        xx_pred = np.argmax(xx_pred, axis=1)
        xx_score.append(f1_score(y_valid, xx_pred, average='weighted'))

        print ("Process X_test ...")
        y_test = clf.predict(X_test)
        y_test = np.argmax(y_test, axis=1)
        
        # 把每一次的结果进行存储
        if index == 0:
            cv_pred = np.array(y_test).reshape(-1, 1)
        else:
            cv_pred = np.hstack((cv_pred, np.array(y_test).reshape(-1, 1)))
    
    # 投票过程
    print(xx_score, np.mean(xx_score))
    submit = []
    for line in cv_pred:
        # np.bincount(line) 见https://blog.csdn.net/xlinsist/article/details/51346523
        # np.argmax(a)返回a中最大数的下表
        submit.append(np.argmax(np.bincount(line)))
    return submit
```

* 这里的结果投票是指，在k-fold训练得到k个分类器后，让每个分类器都对测试数据进行预测得到一个结果。然后选出得票最多的结果作为最终的预测结果。

### 模型stacking

* 大致了解了stack的原理，但是没有实践，整理几篇文章如下

    * https://blog.csdn.net/qq1483661204/article/details/80157365

    * https://zhuanlan.zhihu.com/p/25836678


### 数据观察

此次比赛“数据观察”好像又很大的作用

* 在9.25更换数据之前，有大佬发现net_service=3的样本label基本都是一致的，因此这部分数据几乎可以直接赋值。

* 更换数据之后，在群里看到了“3+8”的预测方式，即根据观察到的数据特别，将数据预测分为两部分，一部分的结果标签只有三类，一部分只有八类，最终把3+8的预测结果进行融合即可。

## 超参选择

* Grid'Search

---

2018.10.19