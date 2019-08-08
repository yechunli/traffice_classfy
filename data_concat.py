import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
#定义oneHot编码器
enc = OneHotEncoder()

def data_concat(csvfile_list, train_list, test_list):
    #文件个数即为业务类型个数，因此定义一个list，用于oneHot编码fit
    array = [x for x in range(len(csvfile_list))]
    #oneHot编码fit需要是2维的
    array = np.reshape(array, newshape=[-1, 1])
    enc.fit(array)
    #用于存储所有文件的数据
    file_data = []
    for index in range(len(csvfile_list)):
        index_file = pd.read_csv(csvfile_list[index])
        #给数据添加label
        index_file['label'] = index
        #把每个文件的数据保存成一个list
        file_data.append(index_file)
    #把所有数据按照列拼接
    all_data = pd.concat(file_data)
    #随机打散所有数据
    shuffle_data = shuffle(all_data)
    #取出数据的label
    y = shuffle_data.pop('label')
    #对label进行oneHot编码，并以arrary的形式保存
    y = enc.transform(np.reshape(y, newshape=[-1,1])).toarray()
    #将y保存为Dataframe格式，其中列名称为0-len(csvfile_list)即文件个数
    y = pd.DataFrame(y, columns=[x for x in range(len(csvfile_list))])
    #其他数据为特征
    x = shuffle_data
    #随机拆分，25%的数据为测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    #数据存入到csv中，后续训练不在需要样本处理过程，index=False表示写入时不写入dataframe的index
    x_train.to_csv(train_list[0], index=False)
    x_test.to_csv(test_list[0], index=False)
    y_train.to_csv(train_list[1], index=False)
    y_test.to_csv(test_list[1], index=False)