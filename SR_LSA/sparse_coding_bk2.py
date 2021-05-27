import numpy as np
import util
import feature_sign_search
import FSS_ref
import dummy_solver
import feature_sign_search_ori
import bases
import lagrange_dual_learn


def sparse_coding(X, num_bases, beta, num_iters, iter_callback=None):
    B_lag = np.random.random((X.shape[0], num_bases)) - 0.5
    # B = B / np.sqrt(np.sum(B**2, 0))

    S_fss = np.zeros((num_bases, X.shape[1]))
    S_dum = np.zeros((num_bases, X.shape[1]))

    for t in range(num_iters):
        # shuffle samples
        np.random.shuffle(X.T)
        print("num_iters:", t)
        for j in range(X.shape[1]):
            # print("t %i sample %i %s" % (t, j, X[:, j]))
            S_fss[:, j] = FSS_ref.feature_sign_search(B_lag, X[:, j], beta)
            # S_dum[:, j] = dummy_solver.feature_sign_search(B, X[:, j], beta)
            # print("t %i coding %i %s" % (t, j, S[:, j]))
        S_fss[np.isnan(S_fss)] = 0
        # S_dum[np.isnan(S_dum)] = 0

        # print(S_fss - S_dum)
        B_lag = lagrange_dual_learn.lagrange_dual_learn(X, S_fss, 1.0)
        # B_fss = dummy_solver.lagrange_dual_learn(X, S_fss, 1.0)
        print("bases done")
        # print(B_fss-B_lag)
        target_value = util.norm_F(X - B_lag @ S_fss) ** 2
        print("target_value = ", target_value)
        # iter_callback(B, S)

    return (B_lag, S_fss)


def SR_testing(X, B, num_bases, beta):
    S = np.zeros((num_bases, X.shape[1]))
    for i in range(X.shape[1]):
        S[:, i] = FSS_ref.feature_sign_search(B, X[:, j], beta)
    return S