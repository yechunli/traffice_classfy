import scapy.all as scapy #不可删除
from scapy.utils import PcapReader
import numpy as np
import csv
from sklearn.preprocessing import MinMaxScaler

# 定义一个规范器
scaler = MinMaxScaler()
#定义一个0-255的列表
normalization = [x for x in range(256)]
#定义把0-255标准化为0-1的标准模型
model = scaler.fit(np.reshape(normalization, newshape=[-1, 1]))

def process(csvfile, filename, DATA_LENGTH):
    #计数
    count = 0
    #定义csv写入器
    writer = csv.writer(csvfile)
    writer.writerow(["data"])
    #定义pcap读入器
    pr = PcapReader(filename)
    #循环一行一行数据读入，直到为空
    while True:
        pcap_data = pr.read_packet()
        #空则退出
        if pcap_data is None:
            break
        else:
            ether_data = np.fromstring(pcap_data.original, np.uint8, count=14)

            type = ether_data[12:14]
            # 去除Ethernet包头，取IP包
            ip_data = pcap_data.payload
            #转换成0-255的数
            ipdata = np.fromstring(ip_data.original, np.uint8)
            #求数据长度
            length = len(ipdata)
            #去IP包头
            communication_data= ip_data.payload
            # 定义一个tmp数据变量，用于存储扩展udp包头与Tcp包头一样长后的udp数据，定义为1500长度，并且填充0，用不补0
            tmp_data = np.zeros(shape=[1500], dtype=np.uint8)

            # 查找UDP包头的pcap
            if communication_data.name == 'UDP':
                #ip包长+udp包头长=28字节，因此取前28
                tmp_data[0:27] = ipdata[0:27]
                #后面补12个0，使其长度与IP+TCP的40字节一致
                #后面接上，UDP包长度超过1488 = 1500 - 12的部分去掉，因为补12个0与TCP包头对齐
                if length > DATA_LENGTH - 12:
                    tmp_data[28:] = ipdata[28:DATA_LENGTH-12]
                else:
                    tmp_data[28:length] = ipdata[28:length]
            else:
                # TCP保持不变，超过1500的部分去掉
                if length > DATA_LENGTH:
                    tmp_data = ipdata[0:DATA_LENGTH]
                else:
                    tmp_data[0:length] = ipdata[0:length]

            #用规范模型对数据进行规范化
            X_train = model.transform(np.reshape(tmp_data, newshape=[-1,1]))
            # 写入时写入一行，因此transpose
            writer.writerow(np.transpose(X_train)[0])
            count = count + 1
            if count == 100:
                break

    # 关闭scapy读入器，防止内存泄露
    pr.close()
    #反馈文件中的数据总数
    return count

# #测试函数用
# #定义数据最大长度1500，不足补0
# DATA_LENGTH = 1500
# #打开csv文件用于写入
# csvfile = open("test.csv", "w")
# # 读取pcap文件
# filename = 'F:\业务分类\sourcecode\datasets\pcap\Game.pcap'
# count = process(csvfile, filename, DATA_LENGTH)
# # 关闭文件
# csvfile.close()
