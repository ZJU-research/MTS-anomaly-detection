import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

var_select_list = [0, 2, 4, 6, 8, 10]


def cnt_data(dataFile):
    data = scio.loadmat(dataFile)
    key = ''
    for i in data.keys():
        key = i
        print(key)
    correct_data = data[key][0][0][1]
    test_data_1 = correct_data[14][0]
    test_data_3 = correct_data[16][0]

    wrong_data = data[key][0][0][3]
    test_data_2 = wrong_data[7][0]

    test_data = np.concatenate((test_data_1, test_data_2, test_data_3), axis=0)
    # test_data = test_data[:, var_select_list]
    plt.plot(test_data)

dataFiles = ['/Users/yuanhanyang/OneDrive/毕设/数据集/2018/MACHINE_DATA']
for dataFile in dataFiles:
    cnt_data(dataFile)

    plt.title("var")
    plt.show()
