import numpy as np
from pyswarms.single.global_best import GlobalBestPSO
from fuzzyCMeansClustering import *
from parameters import *


# def fitness(x):
#     f = fuzzyCMeansClustering(x)
#     return f


def pso():
    # instatiate the optimizer
    x_max = np.ones(n - 1)
    x_min = np.zeros(n - 1)
    bounds = (x_min, x_max)
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    q = 10  # window length
    clusterCenterNumber = 5
    optimizer = GlobalBestPSO(n_particles=10, dimensions=n - 1, options=options, bounds=bounds)

    best_q = q
    best_clusterCenterNumber = clusterCenterNumber
    best_f = -1
    # for _ in range(0):
    #     for _ in range(0):
    #         fcm = fuzzyCmeansClustering(q, clusterCenterNumber)
    #         _, _ = optimizer.optimize(fcm.fuzzyCMeansClustering, 10)
    #         anomaly_score_per_win = fcm.anomaly_score_per_win
    #         f = max(anomaly_score_per_win) / np.mean(anomaly_score_per_win)
    #         if f > best_f:
    #             best_q = q
    #             best_clusterCenterNumber = clusterCenterNumber
    #             best_f = f
    #         clusterCenterNumber += 1
    #     q += 1

    fcm = fuzzyCmeansClustering(best_q, best_clusterCenterNumber)
    cost, pos = optimizer.optimize(fcm.fuzzyCMeansClustering, 3)
    # _ = fcm.anomaly_score_per_win
    # anomaly_score = fcm.cal_anomaly()
    anomaly_score = fcm.anomaly_score_per_win
    return best_q, best_clusterCenterNumber, anomaly_score
