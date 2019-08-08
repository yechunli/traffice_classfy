import pandas as pd
import tensorflow as tf
from model import NN

def train_model(DATA_LENGTH, num_file):
    #文件读取
    x_train = pd.read_csv("x_train.csv")
    x_test = pd.read_csv("x_test.csv")
    y_train = pd.read_csv("y_train.csv")
    y_test = pd.read_csv("y_test.csv")
    #取出dataframe中的值，组成二维矩阵，行index表示样本index，列表示特征index，可以直接输入到算法中
    x_train = x_train.iloc[:, :].values.reshape(-1, DATA_LENGTH)
    x_test = x_test.iloc[:, :].values.reshape(-1, DATA_LENGTH)
    y_train = y_train.iloc[:, :].values.reshape(-1, num_file)
    y_test = y_test.iloc[:, :].values.reshape(-1, num_file)

    fnn_layers = [600, 500, 400, 300, 200, 100, 50, 11]

    my_netural_netweok = NN(fnn_layers=fnn_layers, input_dim=DATA_LENGTH, output_dim=num_file, batch_size=100, act=tf.nn.relu,
                            learning_rate=0.01, keep_rate=0.05)

    init = tf.initialize_all_variables()
    feed_train = {my_netural_netweok.x: x_train,my_netural_netweok.y: y_train, my_netural_netweok.model:'train'}
    feed_test = {my_netural_netweok.x: x_test, my_netural_netweok.y: y_test, my_netural_netweok.model:'test'}
    epcho = 1000
    with tf.Session() as sess:
        sess.run(init)
        for i in range(epcho):
            error, _= sess.run([my_netural_netweok.loss, my_netural_netweok.train_op], feed_dict=feed_train)
            if(i % 10 == 0):
                print('train loss', error)
                acc = sess.run(my_netural_netweok.accuracy, feed_dict=feed_test)
                print('test accuracy', acc)

