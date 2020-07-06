from enum import Enum
import scipy.io as io
import scipy as scp
import numpy as np
import scipy.sparse as sparse
class Method(Enum):
    JACOBI = "Jacobi"
    GAUSS_SEIDEL = "Gauss-Seidel"
    GRADIENT = "Gradient"
    CONJGRADIENT = "Conjugate gradient"


def read_matrix(input_file):
    """
    Reads a matrix from file path. Matrix must be a Matrix Market format.
    Returns either a dense or sparse scipy matrix, depending on file content

    Parameters
    ----------
    input_file: string

    Returns
    ----------
    matrix: sparse matrix


    """

    ##checks for right input type and right format
    if (not type(input_file) == str) or (not input_file.endswith(".mtx")):
        raise Exception("Wrong file extension")

    file = open(input_file)

    #split line into individual strings
    rows, columns, nnz = file.readline().split()

    #convert to int
    total_rows = int(rows)
    total_columns = int(columns)
    nnz = int(nnz)

    rows = np.empty(0)
    columns = np.empty(0)
    values = np.empty(0)
    for line in file:
        #read line, split into individual strings, convert to numbers, append
        element = line.split()

        #-1 required since mtx indexing starts from 1
        rows = np.append(rows, int(element[0])-1)
        columns = np.append(columns, int(element[1])-1)
        values = np.append(values, float(element[2]))

    matrix = sparse.csr_matrix((values, (rows, columns)), shape=(total_rows, total_columns))
    return matrix

def solve_ls(matrix, b, solution, method = Method.JACOBI):
    """
    """
     # Type checking
    if not isinstance(method, Method):
        raise TypeError('Method not supported')

    import ipdb; ipdb.set_trace()
