import tensorflow as tf
import sys
import numpy as np
from trf import trf_linear
from lsmr import lsmr

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

@tf.function
def projZonotope(A, b, n, m):
    #un-ravel
    #A = tf.reshape(A_arg, (n,m))
    #A = A_arg
    ones = np.squeeze(np.asarray(np.ones(A.shape[1], )))
    neg_ones = -1 * ones
    #A_coo = csr_matrix(A)

    A = tf.cast(A, tf.float64)
    b = tf.cast(b, tf.float64)
    b = tf.reshape(b, (tf.size(b), 1))

    x_lsq = tf.linalg.lstsq(A, b)
    #temp = tf.reshape(tf.linalg.diag(x_lsq), [tf.shape(x_lsq)[0]])
    #tf.print("x_lsq", x_lsq, output_stream=sys.stdout)
    #print(x_lsq.shape)
    #eps = bvls(A, b, x_lsq, neg_ones, ones, 1e-13, 200, 2)
    eps = trf_linear(A, b, x_lsq, neg_ones, ones, 1e-13, 'lsmr', 1e-13, 200, 0)

    #NOTE THIS WORKS
    #eps = lsq_linear(A_coo, b, bounds=(neg_ones, ones), lsq_solver='lsmr', lsmr_tol=1e-13, verbose=0).x
    #eps = lsq_linear(A, b, bounds=(neg_ones, ones), lsq_solver='exact', lsmr_tol=1e-13, verbose=0).x
    #eps = tf.py_function(calcLSQ, [A, b], (tf.float64,tf.float64))
    return eps
    #return x_lsq
