import numpy as np
from scipy.sparse import csr_matrix 

from trf_linear import trf_linear
from lsmr import lsmr

# NOTE AND REMINDER TO INCLUDE COPYRIGHT NOTICES

"""
Projects a point onto a zonotope, that is, finds the point
in the zonotope that is closest to the desired point to be
projected. Returns epsilon value in the form:

    A * eps + b' = c

Where A is matrix passed in, and parameter b = c - b'
b' is the center of the zonotope, c is the point to be projected

Utilizes lsq_linear algorithm in scipy.optimize, so there
will be some error in calculation due to forcing matrix 
into a sparse representation to work with scipy.

Parameters
----------
A: matrix (n x m) representing zonotope centered at the origin
b: point in R^n to project onto zonotope A
n: length of matrix A
m: width of matrix A

Returns
-------
eps: point in R^m representing the projection
"""

#@tf.function
def projZonotope(A, b):
    ones = np.squeeze(np.asarray(np.ones(A.shape[1], )))
    neg_ones = -1 * ones

    x_lsq = lsmr(A, b, atol=1e-10, btol=1e-10)[0]
    print("x_lsq", x_lsq)
    eps = trf_linear(A, b, x_lsq, neg_ones, ones, 1e-13, 'lsmr', None, 200, 0)

    return eps

