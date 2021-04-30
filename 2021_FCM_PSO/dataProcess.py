from sklearn import preprocessing
import numpy as np
import wfdb
from parameters import *
import matplotlib.pyplot as plt


class Subsequence:
    __slots__ = ["mat", "group", "membership"]

    def __init__(self, q, clusterCenterNumber):
        self.mat =np.zeros(shape=(n, q))
        self.group = 0
        self.membership = [0.0 for _ in range(clusterCenterNumber)]


def data_processing(q, N, clusterCenterNumber):
    path = '/Users/yuanhanyang/OneDrive/毕设/数据集/2021/mit-bih-arrhythmia-database-1.0.0/100'
    record = wfdb.rdrecord(path, physical=False)
    signal_len = record.sig_len
    data = np.zeros(shape=(int(signal_len / 100), 2))
    for i in range(data.shape[0]):
        data[i, :] = record.d_signal[100 * i, :]
    ref_data = data[0:p, :]

    plt.subplot(211)
    plt.plot(ref_data[:, var_select_list])

    X = ref_data
    X = X[:, var_select_list]
    X = preprocessing.scale(X)
    subseqs = [Subsequence(q, clusterCenterNumber) for _ in range(N)]
    for k in range(N):
        tmp = X[k * r:k * r + q, :]
        subseqs[k].mat = tmp.T
    return subseqs

def onto_unit(x):
    a = np.min(x)
    b = np.max(x)
    return (x - a) / (b - a)