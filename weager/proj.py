import tensorflow as tf
import sys
import numpy as np
from trf import trf_linear

"""
Projects a point onto a zonotope, that is, finds the point
in the zonotope that is closest to the desired point to be
projected. Returns epsilon value in the form:

    A * eps + b' = c

Where A is matrix passed in, and parameter b = c - b'
b' is the center of the zonotope, c is the point to be projected

Since the optimization problem is convex, a found minimum is
guaranteed to be global. Note that the algorithm is not 
guaranteed to project a point back onto itself if that point
lies within the zonotope for this reason.

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
def projZonotope(A, b):
    ones = np.squeeze(np.asarray(np.ones(A.shape[1], )))
    neg_ones = -1 * ones

    A = tf.cast(A, tf.float64)
    b = tf.cast(b, tf.float64)
    b = tf.reshape(b, (tf.size(b), 1))

    x_lsq = tf.linalg.lstsq(A, b, fast=False) # False since Cholesky decomposition occasionally fails
    eps = trf_linear(A, b, x_lsq, neg_ones, ones, 1e-13, 'lsmr', 1e-13, 200, 0)

    return eps
