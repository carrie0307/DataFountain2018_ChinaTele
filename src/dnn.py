import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.contrib import slim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
batch_size=10000
save_model_path='./model/dnn.ckpt'
learning_rate=0.0005
iteration_steps=3

""" downloaded from the QQ group"""

#学习率和reload, is_train, iteration_steps    超参数有正则化参数，学习率，迭代次数
def inference(shape):
    x=tf.placeholder(tf.float64,shape=[batch_size,shape[0]],name='x-input')
    y=tf.placeholder(tf.int64,shape=[batch_size,15],name='y-input')

    regularizer=slim.l2_regularizer(0.004)
    #how to add dropout
    with slim.arg_scope([slim.fully_connected],activation_fn=tf.nn.relu,weights_regularizer=regularizer,
                        biases_regularizer=regularizer,):
        #seven fc layers    [512,1024,1024,1024,1024,1024,15]
        x=slim.fully_connected(x,512,scope='fc1')
        x=slim.fully_connected(x,1024,scope='fc2')
        x=slim.fully_connected(x,1024,scope='fc3')
        x=slim.fully_connected(x,1024,scope='fc4')
        x=slim.fully_connected(x,1024,scope='fc5')
        x=slim.dropout(x,0.5)
        x=slim.fully_connected(x,1024,scope='fc6')
        x=slim.dropout(x,0.5)
        x=slim.fully_connected(x,512,scope='fc7')
        x=slim.dropout(x,0.5)
        x=slim.fully_connected(x,256,scope='fc8')
        x=slim.dropout(x,0.5)
        x=slim.fully_connected(x,15,scope='fc9')
        #print('x shape :',x.get_shape())
        y_pre=slim.softmax(x,scope='softmax')   #y_pre shape is  x by 15
    y_index=tf.argmax(y_pre,axis=1)
    y_index=tf.expand_dims(y_index,axis=1,name='y-index')

    slim.losses.softmax_cross_entropy(x,y)

    total_loss=slim.losses.get_total_loss()
    y_index=tf.cast(y_index,tf.int64)
    y_=tf.argmax(y,axis=1)
    y_=tf.expand_dims(y_,axis=1)
    accuracy=tf.reduce_mean(tf.cast(tf.equal(y_index,y_),tf.float64))
    return total_loss,accuracy

def back_forward(train_data,train_label,total_loss,validation,accuracy,
                 test_data,user_id,label2current_service,reload=False):
    global_step=tf.Variable(0,trainable=False,name='global_step')
    saver=tf.train.Saver(max_to_keep=3)
    train_op=tf.train.AdamOptimizer(learning_rate).minimize(total_loss,global_step=global_step)
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])

        if not os.path.exists(save_model_path):
            os.mkdir(save_model_path)
        if reload==True:
            model_path=tf.train.latest_checkpoint('./model/')
            print('model_path',model_path)
            saver.restore(sess,model_path)
        #train data's type is np.array
        sample_nums=train_data.shape[0]

        x=tf.get_default_graph().get_tensor_by_name('x-input:0')
        y=tf.get_default_graph().get_tensor_by_name('y-input:0')
        for i in range(iteration_steps):
            start,end=0,0
            end+=batch_size
            while(end<=sample_nums):
                _,loss1=sess.run([train_op,total_loss],feed_dict={x:train_data[start:end,:],y:train_label[start:end,:]})
                print('loss',loss1)
                if (end+batch_size)>sample_nums:
                    extra=(end+batch_size)%sample_nums
                    train_data_temp=np.concatenate((train_data[end:sample_nums,:],train_data[0:extra,:]),axis=0)
                    train_label_temp=np.concatenate([train_label[end:sample_nums,:],train_label[0:extra,:]],axis=0)
                    sess.run([train_op],feed_dict={x:train_data_temp,y:train_label_temp})
                start=end
                end+=batch_size
            saver.save(sess, save_model_path,global_step=global_step)
            print(train_label_temp.shape)
            accuracy1=sess.run([accuracy],feed_dict={x:validation[0][:10000,:],y:validation[1][:10000,:]})
            print('%dth iteration accuracy is'%(i),accuracy1)


def preprocess(data,data1):
    feat_one_hot = ['service_type', 'is_mix_service', 'many_over_bill', 'contract_type', 'is_promise_low_consume',
                    'net_service', 'gender']
    columns=data1.columns
    data=data.drop(data[np.isin(data,'\\N')].index)
    data1.loc[data1['2_total_fee']=='\\N','2_total_fee']=data1.loc[data1['2_total_fee']=='\\N','1_total_fee']
    data1.loc[data1['3_total_fee']=='\\N','3_total_fee']=data1.loc[data1['3_total_fee']=='\\N','4_total_fee']
    for feat in feat_one_hot:
        data[feat] = data[feat].map(lambda x: int(x))
        data1[feat] = data1[feat].map(lambda x: int(x))
        train_tem = pd.get_dummies(data[feat])
        train_tem.columns = [feat + str(i) for i in range(len(train_tem.columns))]
        data = pd.concat([data, train_tem], axis=1)
        test_tem = pd.get_dummies(data1[feat])
        test_tem.columns = [feat + str(i) for i in range(len(test_tem.columns))]
        data1 = pd.concat([data1, test_tem], axis=1)
        del (data[feat])
        del (data1[feat])

    feat_normal = [item for item in columns if item not in feat_one_hot]
    sc=StandardScaler()
    #del(data['current_service'])
    train_ = data
    train_[feat_normal] = sc.fit_transform(data[feat_normal])
    test_ = data1
    test_[feat_normal] = sc.fit_transform(data1[feat_normal])
    return train_,test_


def extract_label(train_data):
    #target = pd.get_dummies(train_data['current_service'])
    label=pd.get_dummies(train_data['current_service'])
    train_data.pop('current_service')
    return train_data,label

def main(is_train=True):
    train_data=pd.read_csv('./data/train.csv')
    train_data.pop('user_id')
    test_data=pd.read_csv('./data/test.csv')
    user_id =test_data.pop('user_id')
    label2current_service = dict(
        zip(range(0, len(set(train_data['current_service']))), sorted(list(set(train_data['current_service'])))))
    current_service2label = dict(
        zip(sorted(list(set(train_data['current_service']))), range(0, len(set(train_data['current_service'])))))
    train_data['current_service'] = train_data['current_service'].map(current_service2label)

    processed_train_data,processed_test_data=preprocess(train_data,test_data)   #return dataframe
    processed_train_data,train_label=extract_label(processed_train_data)
    processed_train_data, processed_val_data, train_label, processed_val_label = train_test_split(processed_train_data,
                                                                                 train_label,
                                                                               test_size=0.2, random_state=0)
    processed_train_data=processed_train_data.values
    processed_val_data=processed_val_data.values
    train_label=train_label.values
    processed_val_label=processed_val_label.values
    test_data=processed_test_data.values
    validation=[]
    validation.append(processed_val_data)
    validation.append(processed_val_label)

    shape=[processed_train_data.shape[1],15]
    print('train data shape',shape)

    with tf.Graph().as_default():
        total_loss,accuracy=inference(shape)
        if is_train:
            back_forward(processed_train_data,train_label,total_loss,validation,accuracy,
                         test_data,user_id,label2current_service,reload=True)

        else:
            with tf.Session() as sess:
                saver=tf.train.Saver()
                model_path=tf.train.latest_checkpoint('./model')
                print(model_path)
                sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
                saver.restore(sess, model_path)
                # print('sleeping 10s')
                # time.sleep(10)
                extra=300000-test_data.shape[0]
                test_data=np.vstack((test_data,test_data[:extra,:]))
                ypres=np.empty(shape=(300000,1))
                for i in range(30):
                    ypres1=sess.run([tf.get_default_graph().get_tensor_by_name('y-index:0')],
                                       feed_dict={tf.get_default_graph().get_tensor_by_name('x-input:0'):
                                                      test_data[i*10000:i*10000+10000,:]})[0] #不加[0]是列表,加了是np.nd
                    ypres[i * 10000:i * 10000 + 10000, :] = ypres1


                ypres=ypres[:test_data.shape[0]-extra]
                result=pd.DataFrame()
                result['user_id']=user_id[0:300000]
                result['current_service']=ypres
                result['current_service']=result['current_service'].map(label2current_service)
                result.to_csv('result.csv',index=False)




if __name__=='__main__':
    main(is_train=True)