# # import pyswarms as ps
# # from pyswarms.utils.functions import single_obj as fx
# #
# # # Set-up hyperparameters
# # options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
# #
# # def fitness(x):
# #     return x[:, 0]*2 + x[:, 1]*2
# #
# #
# # # Call instance of GlobalBestPSO
# # optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2,
# #                                     options=options)
# #
# # # Perform optimization
# # stats = optimizer.optimize(fitness(), iters=100)
#
# # import modules
# import numpy as np
#
#
# def fitness(x):
#     return -(x[:, 0] ** 2 + x[:, 1] ** 2)
#
#
# # create a parameterized version of the classic Rosenbrock unconstrained optimzation function
# def rosenbrock_with_args(x):
#     f = fitness(x)
#     return f
#
#
# from pyswarms.single.global_best import GlobalBestPSO
#
# # instatiate the optimizer
# x_max = 10 * np.ones(2)
# x_min = -1 * x_max
# bounds = (x_min, x_max)
# options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
# optimizer = GlobalBestPSO(n_particles=10, dimensions=2, options=options, bounds=bounds)
#
# # now run the optimization, pass a=1 and b=100 as a tuple assigned to args
#
# cost, pos = optimizer.optimize(rosenbrock_with_args, 1000)

import numpy as np

a = np.ones(shape=(2, 3))
print(a)
b = 3
print(a / b)
