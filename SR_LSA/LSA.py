import numpy as np
from sklearn.decomposition import TruncatedSVD


def LSA_training(F_ref):
    # 计算参数K
    n_c = min(F_ref.shape[0], F_ref.shape[1]) - 1
    print("F_ref's shape: %d %d" % (F_ref.shape[0], F_ref.shape[1]))
    svd = TruncatedSVD(n_components=n_c, random_state=42)
    svd.fit(F_ref)
    # print(svd.singular_values_)
    # print(svd.explained_variance_ratio_)
    total_explained = 0
    k = None
    for i in range(len(svd.explained_variance_ratio_)):
        total_explained = total_explained + svd.explained_variance_ratio_[i]
        if total_explained >= 0.9:
            k = i + 1
            break
    print(" k = ", k)

    # 计算u_kref; S_kref
    u_ref, s_ref, v_ref = np.linalg.svd(F_ref, full_matrices=False)
    # print(u_ref)
    # print(s_ref)
    # print(v_ref)
    # print(np.dot(u_ref, np.dot(np.diag(s_ref), v_ref)))
    u_Kref = u_ref[:, :k]
    s_kref = np.diag(s_ref[:k])
    v_Kref = v_ref[:k, :]
    return u_Kref, s_kref, v_Kref


def LSA_testing(F_test, u_kref, s_kref):
    print("F_test's shape: %d %d" % (F_test.shape[0], F_test.shape[1]))
    F_hat = np.dot(np.dot(np.linalg.pinv(s_kref), u_kref.T), F_test)
    F_recon = np.dot(np.dot(u_kref, s_kref), F_hat)
    anomaly_score = (F_test - F_recon) * (F_test - F_recon)
    return anomaly_score
