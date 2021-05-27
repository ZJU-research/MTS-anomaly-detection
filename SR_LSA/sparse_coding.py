import numpy as np
import util
import feature_sign_search
import FSS_ref
import dummy_solver

def sparse_coding(X, num_bases, gamma, num_iters, lamda, iter_callback=None):
    B = np.random.random((X.shape[0], num_bases)) - 0.5
    B = B / np.sqrt(np.sum(B**2, 0))
    # B = ortho_group.rvs(dim=num_bases)
    # B = B[0:X.shape[0], :]
    S = np.zeros((num_bases, X.shape[1]))

    for t in range(num_iters):
        # shuffle samples
        # np.random.shuffle(X.T)
        print("num_iters:", t)
        for j in range(X.shape[1]):
            # print("t %i sample %i %s" % (t, j, X[:, j]))
            S[:, j] = FSS_ref.feature_sign_search(B, X[:, j], gamma)
            # print("t %i coding %i %s" % (t, j, S[:, j]))
        S[np.isnan(S)] = 0

        B = lagrange_dual_learn.lagrange_dual_learn(X, S, lamda)
        print("bases done")
        target_value = util.norm_F(X - B @ S) ** 2
        print("target_value = ", target_value)
        # iter_callback(B, S)

    return (S, B)


def SR_testing(X, B, num_bases, gamma):
    S = np.zeros((num_bases, X.shape[1]))
    for i in range(X.shape[1]):
        S[:, i] = FSS_ref.feature_sign_search(B, X[:, i], gamma)
    return S