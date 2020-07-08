import iterative
import numpy as np
matrix = iterative.read_matrix("vem2.mtx")
x = np.ones(shape = (matrix.shape[0], 1))
b = matrix @ x
iterative.validate(matrix, b, 0.00001, x)
