import iterative
import numpy as np
import os
import time

for filename in os.listdir(os.getcwd()):
    if filename.endswith(".mtx"):
        print("Matrix:" + filename)
        matrix = iterative.read_matrix(filename)
        x = np.ones(shape = (matrix.shape[0], 1))
        b = matrix @ x
        for tol in [10e-4, 10e-6, 10e-8, 10e-10]:

            print("\n\nTolerance: " + str(tol))
            iterative.validate(matrix, b, tol, x)
        print("\n\n")

input("Press key to exit")
