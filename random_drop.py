import pandas as pd
import random

def random_drop(file_list, total_num, drop_num):#文件列表，每个文件的总数据数，每个文件要丢弃的数据数
    #对每一个文件的数据进行随机丢弃
    for index in range(len(file_list)):
        #生成drop_num个随机数，这些数范围是从1到total_num，random_num是一个列表
        random_num = random.sample(range(1, total_num[index]), drop_num[index])
        #读取文件
        samples = pd.read_csv(file_list[index])
        #丢弃random_num列表中的数对应的行
        samples.drop(random_num, inplace=True)
        #写回文件中
        samples.to_csv(file_list[index])

