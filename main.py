from data_process import process
from random_drop import random_drop
from data_concat import data_concat
from train_model import train_model
#from model import NN
#import tensorflow as tf
#import pandas as pd
original_dir = 'D:\资料\业务分类\李刚的代码\datasets\pcap\\'
filename_list = [original_dir + 'Game.pcap',
                 original_dir + 'gftp10M_v2.pcap',
                 original_dir + 'gmailchat3.pcap',
                 original_dir + 'live.pcap',
                 original_dir + 'Music_z.pcap',
                 original_dir + 'pcap_HLS.pcap',
                 original_dir + 'pcap_VR_10000.pcap',
                 original_dir + 'SP_JD_z.pcap',
                 original_dir + 'VideoCall_z.pcap',
                 original_dir + 'VOIP_z.pcap',
                 original_dir + 'Web_z.pcap',
                 original_dir + 'WeChat_z.pcap']
nomalized_data = 'F:\\'
csvfile_list = [nomalized_data + 'game.csv',
                nomalized_data + 'gftp.csv',
                nomalized_data + 'gmailchat.csv',
                nomalized_data + 'live.csv',
                nomalized_data + 'music.csv',
                nomalized_data + 'hls.csv,',
                nomalized_data + 'vr.csv',
                nomalized_data + 'jd.csv',
                nomalized_data + 'video.csv',
                nomalized_data + 'voip.csv',
                nomalized_data + 'web.csv',
                nomalized_data + 'wechat.csv']

# train_data = 'train_data/'
# train_list = [train_data + 'x_train.csv',
#               train_data + 'y_train.csv']
# test_data = 'test_data/'
# test_list = [test_data + 'x_test.csv',
#              test_data + 'y_test.csv']

num_file = len(csvfile_list)
#数据集文件列表
#filename_list = ['F:\业务分类\sourcecode\datasets\pcap\Game.pcap']
#保存处理后的数据集的文件
#csvfile_list = ['game.csv']
#每条数据的长度
DATA_LENGTH = 1500
#存储每个数据集文件中数据的个数
num_list = []


for index in range(num_file):
    #w表示写入，newline=“”表示两行写入之间不空行
    with open(csvfile_list[index], 'w', newline="") as file:
        #将数据截取为1500字节长度，其中UDP包头长度后补12个0，与TCP对齐，并把每字节数据换算成0-255之间的数，并作0-1标准化
        #最后写入csvfile_list中对应的文件中
        num = process(file, filename_list[index], DATA_LENGTH)
        num_list.append(num)


#选出数据个数最少的
min_num = min(num_list)
#计算丢弃个数
drop_num = [x - min_num for x in num_list]
#选取数据个数最少的，把比最少的数据集个数多的数据集中多出的部分全部随机丢弃
random_drop(csvfile_list, num_list, drop_num)
#对多个文件的数据concat，随机拆分为训练集和测试集，并添加label
data_concat(csvfile_list, train_list, test_list)

train_model(DATA_LENGTH, num_file)
# import numpy as np
# print(x_train)
# print(np.shape(x_train))
# print("~~~~~~~~~~~~~~~~~~~~~~~~")
# print(x_test)
# print(np.shape(x_test))
# print("~~~~~~~~~~~~~~~~~~~~~~~~")
# print(y_test)
# print(np.shape(y_test))
# print("~~~~~~~~~~~~~~~~~~~~~~~~")
# print(y_train)
# print(np.shape(y_train))
