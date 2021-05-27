import random
import math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sparse_coding as sc
import LSA
from sklearn import preprocessing

# parameters
window_len = 5
var_select_list = [0, 2, 4, 6, 8, 10]
num_bases = 12
beta = 2


def onto_unit(x):
    a = np.min(x)
    b = np.max(x)
    return (x - a) / (b - a)


def data_processing(X, w):
    X = X[:, var_select_list]
    # X = preprocessing.scale(X)
    for j in range(X.shape[1]):
        X[:, j] = onto_unit(X[:, j])
    n = X.shape[0] - w + 1
    print(X.shape)
    Y = []
    for k in range(X.shape[1]):
        S_k = np.zeros(shape=(w, n))
        for j in range(S_k.shape[1]):
            S_k[:, j] = X[j:j + w, k]
        Y.append(S_k)
    return Y


if __name__ == '__main__':
    data = scipy.io.loadmat('/Users/yuanhanyang/OneDrive/毕设/数据集/2018/MACHINE_DATA')
    key = ''
    for i in data.keys():
        key = i
    print(key)
    correct_data = data[key][0][0][1]
    ref_data = correct_data[0][0]
    test_data_1 = correct_data[14][0]
    test_data_3 = correct_data[16][0]

    wrong_data = data[key][0][0][3]
    test_data_2 = wrong_data[0][0]

    test_data = np.concatenate((test_data_1, test_data_2, test_data_3), axis=0)
    print("ref_data", ref_data.shape)
    print("test_data", test_data.shape)

    # 训练过程
    print("---------------- training phase ----------------")
    Y_ref = data_processing(ref_data, window_len)
    X_ref = []
    B_ref = []
    for S_k in Y_ref:
        print("---------------- processing ----------------")
        # X_kref, B_kref = sc.sparse_coding(S_k, num_bases, 2, 10, lambda B, S: callback(S_k, B, S))
        X_kref, B_kref = sc.sparse_coding(S_k, num_bases, 2, 10)
        X_ref.append(X_kref)
        B_ref.append(B_kref)

    F_ref = None
    for i in range(len(X_ref)):
        if F_ref is None:
            F_ref = X_ref[i]
        else:
            F_ref = np.concatenate((F_ref, X_ref[i]), axis=0)

    u_Kref, s_kref, v_Kref = LSA.LSA_training(F_ref)

    # 测试
    print("---------------- testing phase ----------------")
    Y_test = data_processing(test_data, window_len)
    X_test = []
    for i in range(len(Y_test)):
        X_ktest = sc.SR_testing(Y_test[i], B_ref[i], num_bases, beta)
        X_test.append(X_ktest)

    F_test = None
    for i in range(len(X_test)):
        if F_test is None:
            F_test = X_test[i]
        else:
            F_test = np.concatenate((F_test, X_test[i]), axis=0)

    anomaly_score_m = LSA.LSA_testing(F_test, u_Kref, s_kref)
    print(anomaly_score_m.shape)
    anomaly_score_per_win = np.sum(anomaly_score_m, axis=0)
    anomaly_score = np.zeros(shape=(test_data.shape[0]))
    for i in range(test_data.shape[0]):
        if i < window_len:
            anomaly_score[i] = sum([(anomaly_score_per_win[j]) for j in range(0, i + 1)]) / (i + 1)
        elif window_len <= i <= test_data.shape[0] - window_len:
            anomaly_score[i] = sum([(anomaly_score_per_win[j]) for j in range(i - window_len + 1, i + 1)]) / window_len
        else:
            anomaly_score[i] = sum([(anomaly_score_per_win[j]) for j in range(i - window_len + 1, test_data.shape[0] -
                                                                              window_len + 1)]) / (test_data.shape[0] - i)
    plt.subplot(211)
    plt.plot(test_data)
    plt.set_loglevel("info")
    plt.subplot(212)
    plt.plot(anomaly_score)
    plt.show()
