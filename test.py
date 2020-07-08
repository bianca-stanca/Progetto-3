import iterative
import numpy as np
spa1 = iterative.read_dense_matrix("spa1.mtx")
x = np.ones(shape = (1000, 1))
b = spa1 @ x
iterative.validate(spa1, b, 0.00001, x)
