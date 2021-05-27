'''
python implementation of the algorithms described in the paper
"Efficient sparse coding algorithms"
by Honglak Lee, Alexis Battle, Rajat Raina, and Andrew Y. Ng

for Math 651 at the University of Michigan, Winter 2020

by Michael Ivanitskiy and Vignesh Jagathese
'''

from util import *

def feature_sign_search(A, y, gamma, x0=None):
    dim_m, dim_p = A.shape
    y = np.reshape(y, (-1, 1))
    x = np.zeros((dim_p, 1))
    if x0 is not None:
        x = x0
    theta = np.zeros((dim_p, 1), dtype=np.int8)
    active_set = set()

    def FS_unconstrained_QP_factory(M_hat, sgn_vec):
        return lambda x_vec: norm_2(y - M_hat @ x_vec) ** 2 + gamma * sgn_vec.T @ x_vec

    grad = - 2 * A.T @ y

    # gradient computation function
    def comp_grad():
        return -2 * A.T @ (y - A @ x)

    opcond_b = np.inf

    while opcond_b > gamma:
        print("opcond_b_loop")

        # * 2: select i
        # from zero coefficients of x, select i such that
        # y - A x is changing most rapidly with respect to x_i

        # acivate x_i if it locally improves objective
        grad = comp_grad()
        i_sel = np.argmax(np.abs(grad) * (theta == 0))
        i_deriv = grad[i_sel]

        print('SELECTING:\t%d\t%f' % (i_sel, i_deriv))

        if i_deriv > gamma:
            theta[i_sel] = -1
            x[i_sel] = 0.0
            active_set.add(i_sel)
        elif i_deriv < - gamma:
            theta[i_sel] = 1
            x[i_sel] = 0.0
            active_set.add(i_sel)
        else:
            break

        while True:
            print("opcond_a_loop")

            active_list = np.array(sorted(list(active_set)))
            print(active_list)

            # * 3: feature-sign step
            # A_hat is submatrix of A containing only columns corresponding to active set

            A_hat = np.delete(A, [i for i in range(dim_p) if i not in active_set], 1)
            x_hat = select_elts(x, active_list)
            theta_hat = select_elts(theta, active_list)
            print("x_hat = ", x_hat)
            # compute solution to unconstrained QP:
            # minimize_{x_hat} || y - A_hat @ x_hat ||^2 + gamma * theta_hat.T @ x_hat
            # REVIEW: do we /really/ need to compute matrix inverse? can we minimize or at least compute inverse more efficiently?

            x_hat_new = (
                    np.linalg.inv(A_hat.T @ A_hat)
                    @
                    (A_hat.T @ y - gamma * theta_hat / 2)
            )
            print("X_hat_new = ", x_hat_new.T)

            line_search_func = FS_unconstrained_QP_factory(A_hat, theta_hat)

            x_hat = coeff_discrete_line_search(x_hat, x_hat_new, line_search_func)
            print("X_hat = ", x_hat.T)

            for idx_activ in range(len(active_list)):
                if is_zero(x_hat[idx_activ]):
                    x[active_list[idx_activ]] = 0
                else:
                    x[active_list[idx_activ]] = x_hat[idx_activ]

            for idx_rem in range(len(x_hat)):
                tmp = x_hat[idx_rem]
                if is_zero(tmp):
                    active_set.discard(active_list[idx_rem])

            theta = sign(x)
            grad = comp_grad()
            opcond_a = abs(grad[theta != 0] + gamma * theta[theta != 0])
            print("opcond_a = ", opcond_a)
            if np.allclose(opcond_a, 0):
                break

        grad = comp_grad()
        opcond_b = np.max(abs(grad[theta == 0]))
        print("opcond_b=", opcond_b)

    return x.reshape(-1)
