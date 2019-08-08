"""
this is used to classify application types

2018-9-16: two types
1) HLS video streaming
2) Cloud VR
----------------------

Developed by goodluckfrank
2018-9-10

"""

import numpy as np
from keras.utils import np_utils
from keras.models import Model,Sequential
from keras.layers import Dense, Activation, Convolution1D, AveragePooling1D, Flatten, Dropout
from keras.optimizers import RMSprop, Adam

import scapy.all as scapy
import matplotlib.pyplot as plt

# grid search
#from sklearn.grid_search import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


# STEPS:10-100  length:100:100:1500
#业务类型：1.wechat:videocall voip message
#         2. 网页访问
#         3. 网购：淘宝 京东
#         4.音乐：网易 qq音乐
#         5.游戏 

# STEPS = 80
# INPUT_LENGTH = 1500
# STEPS_arr0=np.arange(5,11,1)
# STEPS_arr=np.append(STEPS_arr0,[40,70,100])
STEPS_arr1=np.arange(10,110,10)
STEPS_arr=np.append([7,8,9],STEPS_arr1)
print(STEPS_arr)
S_len=STEPS_arr.size
INPUT_LENGTH_arr=np.arange(100,1700,200)
print(INPUT_LENGTH_arr)
IL_len=INPUT_LENGTH_arr.size

# result
# emp_arr=np.zeros((S_len,IL_len))
loss_dict={'HLS':np.zeros((S_len,IL_len)),'VR':np.zeros((S_len,IL_len)),'LIVE':np.zeros((S_len,IL_len)),'FTP':np.zeros((S_len,IL_len)),
           'JD':np.zeros((S_len,IL_len)),'VideoCall':np.zeros((S_len,IL_len)),'VOIP':np.zeros((S_len,IL_len)),'Music':np.zeros((S_len,IL_len)),
           'Web':np.zeros((S_len,IL_len)),'WeChat':np.zeros((S_len,IL_len)),'Game':np.zeros((S_len,IL_len))}
acc_dict={'HLS':np.zeros((S_len,IL_len)),'VR':np.zeros((S_len,IL_len)),'LIVE':np.zeros((S_len,IL_len)),'FTP':np.zeros((S_len,IL_len)),
           'JD':np.zeros((S_len,IL_len)),'VideoCall':np.zeros((S_len,IL_len)),'VOIP':np.zeros((S_len,IL_len)),'Music':np.zeros((S_len,IL_len)),
           'Web':np.zeros((S_len,IL_len)),'WeChat':np.zeros((S_len,IL_len)),'Game':np.zeros((S_len,IL_len))}
doc=open('/usr/zhangcong/pythonfiles/out.txt','w')

for i in range(S_len):
    for j in range(IL_len):
        STEPS = STEPS_arr[i]
        INPUT_LENGTH = INPUT_LENGTH_arr[j]

        Mode = "Normal"

        # app_type_dict = { 'HLS': 0, 'VR':1, 'LIVE':2, 'GMC':3,'FTP':4}
        # app_type_dict = { 'HLS': 0, 'VR':1, 'LIVE':2,'FTP':3}
        app_type_dict = {'HLS':0, 'VR':1, 'LIVE':2,'FTP':3,'JD':4,'VideoCall':5,'VOIP':6,'Music':7,'Web':8,'WeChat':9,'Game':10}

        prep_mode = ['Normal','CV']

        file_name = { 'HLS': "/usr/zhangcong/pythonfiles/pcap/pcap_HLS.pcap",
                            'VR':"/usr/zhangcong/pythonfiles/pcap/pcap_VR_10000.pcap",
                                'LIVE':"/usr/zhangcong/pythonfiles/pcap/live.pcap",
                                    'FTP': "/usr/zhangcong/pythonfiles/pcap/gftp10M_v2.pcap",
                                        'JD':"/usr/zhangcong/pythonfiles/pcap/SP_JD_z.pcap",
                                            'VideoCall':"/usr/zhangcong/pythonfiles/pcap/VideoCall_z.pcap",
                                                'VOIP':"/usr/zhangcong/pythonfiles/pcap/VOIP_z.pcap",
                                                    'Music':"/usr/zhangcong/pythonfiles/pcap/Music_z.pcap",
                                                        'Web':"/usr/zhangcong/pythonfiles/pcap/Web_z.pcap",
                                                            'WeChat':"/usr/zhangcong/pythonfiles/pcap/WeChat_z.pcap",
                                                                'Game':"/usr/zhangcong/pythonfiles/pcap/Game.pcap"

            }

        NUM_APP_TYPES = app_type_dict.__len__()


        # fix random seed for reproducibility
        seed = 7
        np.random.seed(seed)


        def pcap_data_preprocessing(filename, app_type, mode):

            pcaps = scapy.rdpcap(filename)

            sample_num = len(pcaps)
            tranportpktlist = np.zeros((sample_num, INPUT_LENGTH), dtype=np.uint8)

            for index in range(sample_num):


                temp_load = pcaps[index].payload.payload.payload

                len_temp_load = len(temp_load)

                low_8_bits = len_temp_load >> 8
                high_8_bits = len_temp_load & 0xff

                tranportpktlist[index, 0] = low_8_bits
                tranportpktlist[index, 1] = high_8_bits

                if(len_temp_load !=0) :

                # temp_load = pcaps[index].payload.payload.payload.original

                    temp_load = pcaps[index].payload.payload.original
                    len_temp_load = len(temp_load)

                    if len_temp_load < INPUT_LENGTH - 2 :
                        tranportpktlist[index, 2:len_temp_load+2] = np.fromstring(temp_load, dtype=np.uint8)
                    else:
                        tranportpktlist[index, 2:INPUT_LENGTH] = np.fromstring(temp_load, dtype=np.uint8)[0:INPUT_LENGTH-2]

            current_app_type = app_type_dict[app_type]

            new_sample_num = int(sample_num / STEPS) * STEPS

            print("%s pcaps length = : %i" % (app_type, new_sample_num))

            # generate x_train_data, y_train_data

            X_data = tranportpktlist[0:new_sample_num, :]

            if (mode == "Normal"):
                from sklearn.model_selection import train_test_split
                X_train, X_test = train_test_split(X_data, test_size=0.2, random_state=0)

                # 分割后重新整理X_train and X_test的长度
                new_X_train_num = int(X_train.shape[0] / STEPS) * STEPS
                new_X_test_num = int(X_test.shape[0] / STEPS) * STEPS

                X_train = X_train[0:new_X_train_num, :]
                X_test = X_test[0:new_X_test_num, :]

                from sklearn.preprocessing import MinMaxScaler
                mm_X = MinMaxScaler()
                X_train = mm_X.fit_transform(X_train)
                X_test = mm_X.transform(X_test)

                X_train = X_train.reshape(-1, STEPS, INPUT_LENGTH)
                X_test = X_test.reshape(-1, STEPS, INPUT_LENGTH)

                # generate y_train and y_test
                y_train = np.zeros(X_train.shape[0], dtype=np.uint8)
                for i in range(X_train.shape[0]):
                    y_train[i] = current_app_type

                y_test = np.zeros(X_test.shape[0], dtype=np.uint8)
                for i in range(X_test.shape[0]):
                    y_test[i] = current_app_type

                y_train = np_utils.to_categorical(y_train, num_classes=NUM_APP_TYPES)
                y_test = np_utils.to_categorical(y_test, num_classes=NUM_APP_TYPES)

                return X_train, y_train, X_test, y_test

            if (mode == "CV"):

                from sklearn.preprocessing import MinMaxScaler
                mm_X = MinMaxScaler()
                X_train = mm_X.fit_transform(X_data)

                X_train = X_train.reshape(-1, STEPS, INPUT_LENGTH)

                # generate y_train and y_test
                y_train = np.zeros(X_train.shape[0], dtype=np.uint8)
                for i in range(X_train.shape[0]):
                    y_train[i] = current_app_type

                y_train = np_utils.to_categorical(y_train, num_classes=NUM_APP_TYPES)

                return X_train, y_train


        def create_model(optimizer ='Adam', dropout_rate = 0.25 ):
            # Create CNN network
            model = Sequential()

            # the first Conv layer 1 output shape ( 200 * input_length)
            model.add(Convolution1D(
                filters=200,
                kernel_size=4,
                strides=1,
                input_shape=(STEPS, INPUT_LENGTH),
            ))
            model.add(Activation('relu'))

            # the second Conv layer 1 output shape ( 80 * input_length)
            model.add(Convolution1D(
                filters=80,
                kernel_size=3,
                strides=1,
            ))
            model.add(Activation('relu'))

            # Average pooling layer, output shape ( 80 * input_length/2)
            model.add(AveragePooling1D(
                pool_size=2,
                strides=2,
                padding='valid',  # Padding method
            ))

            model.add(Dropout(dropout_rate))

            # Fully connected layer 1 input shape ( 80 * input_length/2), output shape (1024)

            model.add(Flatten())

            model.add(Dense(1000, activation="relu"))
            model.add(Dense(500, activation="relu"))
            model.add(Dense(200, activation="relu"))
            model.add(Dense(100, activation="relu"))
            model.add(Dense(50, activation="relu"))
            model.add(Dense(10, activation="relu"))

            # Fully connected layer 2 to shape (NUM_APP_TYPES) for NUM_APP_TYPES classes
            model.add(Dense(NUM_APP_TYPES, activation="softmax"))

            # Another way to define your optimizer
            if(optimizer == 'Adam'):
                optimizer = Adam(lr=1e-4)

            # We add metrics to get more results you want to see
            model.compile(optimizer=optimizer,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

            #print(model.summary())
            for i, layer in enumerate(model.layers):
                print(i, layer.name)

            return model


        # data preprocessing
        X_test_data_dict = {}
        y_test_data_dict = {}

        X_train_data =[]
        y_train_data =[]

        looptime=0
        for key in app_type_dict:

            if (Mode == "Normal"):
                X_train, y_train, X_test, y_test = pcap_data_preprocessing(file_name[key], key, "Normal")
                X_test_data_dict.update({key: X_test})
                y_test_data_dict.update({key: y_test})
            else :  # Mode == "CV"
                X_train, y_train = pcap_data_preprocessing(file_name[key], key, "CV")

            if (looptime == 0) :
                X_train_data =  X_train
                y_train_data =  y_train
                looptime=looptime+1
            else :
                X_train_data = np.concatenate((X_train_data, X_train ), axis=0)
                y_train_data = np.concatenate((y_train_data, y_train ), axis=0)


        if (Mode == "Normal" ):

            # how to set epochs?  wait until the training result is converged
            from keras.callbacks import EarlyStopping

            early_stopping = EarlyStopping(monitor='acc', patience=2)  # val_loss or val_acc

            num_epchs = 100

            model = create_model(dropout_rate = 0.1)
            # validation_split=0.2 : used to split training and testing datasets
            hist = model.fit(X_train_data, y_train_data, epochs=num_epchs, batch_size=32, callbacks=[early_stopping])
            print(hist.history)

            for key in app_type_dict:
                print("\nTesting  for %s------------" % (key))
                # Evaluate the model with the metrics we defined earlier
                # loss_HLS, accuracy_HLS = model.test_on_batch(x_test_data_hls, y_test_data_hls)
                loss, accuracy = model.evaluate(X_test_data_dict[key], y_test_data_dict[key])

                print("\ntest loss of %s: %f" % (key, loss))
                print("\ntest accuracy of %s: %f" % (key, accuracy))

                loss_temp_arr=loss_dict[key]
                loss_temp_arr[i,j]=loss
                # loss_dict[key]=loss_temp_arr

                acc_temp_arr=acc_dict[key]
                acc_temp_arr[i,j]=accuracy
                # acc_dict[key]=acc_temp_arr

print(loss_dict,file=doc)
print(acc_dict,file=doc)
doc.close()

'''
# figure
plt.figure()
# color_dict={ 'HLS': 'r', 'VR':'b', 'LIVE':'k','FTP':'y'}
for key in app_type_dict:
    
    acc_pl_arr=np.array(acc_dict[key])
    # plt.plot(STEPS_arr,acc_pl_arr[:,1],color_dict[key],label=key)
    plt.subplot(121)
    plt.plot(STEPS_arr,acc_pl_arr[:,0],label=key)
    plt.legend()

    plt.subplot(122)
    plt.plot(STEPS_arr,acc_pl_arr[:,1],label=key)
    plt.legend()
plt.show()
'''
