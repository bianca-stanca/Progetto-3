from enum import Enum
import scipy.io as io
import scipy as scp
import numpy as np
import scipy.sparse as sparse

#global variable
MAX_ITER = 30000

class Method(Enum):
    JACOBI = "JACOBI"
    GAUSS_SEIDEL = "GAUSS_SEIDEL"
    GRADIENT = "GRADIENT"
    CONJ_GRADIENT = "CONJ_GRADIENT"


def read_sparse_matrix(input_file):
    """
    Reads a matrix from file path. Matrix must be a Matrix Market format.
    Returns a sparse scipy matrix

    Parameters
    ----------
    input_file: string

    Returns
    ----------
    matrix: sparse matrix


    """

    matrix = sparse.csr_matrix(read_dense_matrix(input_file))

    return matrix

def read_dense_matrix(input_file):
    """
    Reads a matrix from file path. Matrix must be a Matrix Market format.
    Returns a ndarray

    Parameters
    ----------
    input_file: string

    Returns
    ----------
    matrix: ndarray


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

    #initialise empty matrix
    matrix = np.zeros(shape=(total_rows, total_columns))
    for line in file:
        #read line, split into individual strings, convert to numbers, append
        element = line.split()

        #-1 required since mtx indexing starts from 1
        row = int(element[0])-1
        column = int(element[1])-1
        value = float(element[2])

        matrix[row, column] = value

    return matrix

def solve_ls(matrix, b, tol, method = Method.JACOBI):
    """
    """
     # Type checking
    if method not in Method._member_names_ and not isinstance(method, Method):
        raise TypeError('Method not supported')

    #turn b into column array if not already in that shape
    if b.shape[1] != 1:
        np.reshape(b, (b.shape[1], b.shape[0]))
    #initialise first solution to 0 vector
    x = np.zeros((matrix.shape[0], 1))

    #initialise counter
    k = 0

    #get initial error
    termine_noto = b
    residue = b - matrix @ x

    #needed for conjugated gradient
    d = residue
    y = np.zeros_like(residue)

    error = np.linalg.norm(residue) / np.linalg.norm(b)


    diagonal_p = matrix.diagonal()
    inverse_diagonal = np.reciprocal(diagonal_p)

    #reshape into column array
    inverse_diagonal = np.reshape(inverse_diagonal, (inverse_diagonal.shape[0], 1))

    #create lower triangular matrix
    triangular_p = scp.sparse.tril(matrix)

    #sparse.tril always return a sparse matrix. In the case of dense matrixes,
    #convert the triangular matrix to dense in order to keep consistency when
    #measuring performance.
    if type(matrix) == np.ndarray:
        triangular_p = triangular_p.toarray()

    while k < MAX_ITER and error > tol:
        if method == Method.JACOBI or method.upper() == Method.JACOBI.value:
            add_on = update_jacobi(x, residue, inverse_diagonal)
            x = x + add_on

        elif method == Method.GAUSS_SEIDEL or method.upper() == Method.GAUSS_SEIDEL.value:
            add_on = update_gauss(residue, triangular_p)
            x = x + add_on

        elif method == Method.GRADIENT or method.upper == Method.GRADIENT.value:
            alpha = gradient_alpha(matrix, residue)
            x = x + alpha*residue

        elif method == Method.CONJ_GRADIENT or method.upper() == Method.CONJ_GRADIENT.value :
            alpha, y = conjugated_gradient_alpha(x, matrix, residue, d)
            x = x + alpha*d

        residue = b - matrix @ x
        if method == Method.CONJ_GRADIENT or method.upper() == Method.CONJ_GRADIENT.value:
            d = update_conjugated_gradient_d(matrix, residue, d, y)

        #update stopping criterion
        k += 1
        try:
            error = np.linalg.norm(residue) / np.linalg.norm(b)
        except error:
            import ipdb; ipdb.set_trace()
    return x, k

def validate(matrix, b, tol, exact_solution):
    results = np.zeros(shape=(matrix.shape[0], 4))
    iterations = np.zeros(shape=(4))
    execution_time = np.zeros(shape=(4))

    for method, i in zip(Method._member_names_, range(4)):
        print(method)
        result, iterations[i] = solve_ls(matrix, b, tol, method=method)
        results[:, 0] = result[:, 0]

def update_jacobi(x, residue, p_1):
    #elementwise multiplication
    add_on = np.multiply(p_1, residue)
    return add_on

def update_gauss(residue, triangular_p):

    y = forward_substitution(triangular_p, residue)
    return y

def forward_substitution(matrix, b):

    x = np.zeros(shape = (matrix.shape[0], 1))
    if matrix[0,0] == 0:
        raise Exception("Unsolvable linear system")
    x[0] = b[0]/matrix[0,0]

    for i in np.arange(1, matrix.shape[0]):
        if matrix[i, i] == 0:
            raise Exception("Unsolvable linear system")
        x[i] = (b[i]-matrix[i, :] @ x) / matrix[i, i]
    return x

def gradient_alpha(matrix, residue):

    transposed_residue = residue.transpose()
    y = matrix @ residue
    a = transposed_residue @ residue
    b = residue @ y
    return a/b

def conjugated_gradient_alpha(x, matrix, residue, d):

    y = matrix @ d
    z = matrix @ residue


    #returns an array of array
    alpha = (d.transpose() @ residue) / (d.transpose() @ y)
    return alpha[0, 0], y

def update_conjugated_gradient_d(matrix, residue, d, y):
    w = matrix @ residue

    #returns an array of an array
    beta = (d.transpose() @ w) / (d.transpose() @ y)

    return residue - beta[0, 0] * d
