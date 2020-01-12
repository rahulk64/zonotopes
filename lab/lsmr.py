import numpy as np
import sys
import tensorflow as tf

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

from numpy import zeros, infty, atleast_1d, result_type
from numpy.linalg import norm
from math import sqrt

def _sym_ortho(a, b):
    """
    Stable implementation of Givens rotation.
    Notes
    -----
    The routine 'SymOrtho' was added for numerical stability. This is
    recommended by S.-C. Choi in [1]_.  It removes the unpleasant potential of
    ``1/eps`` in some important places (see, for example text following
    "Compute the next plane rotation Qk" in minres.py).
    References
    ----------
    .. [1] S.-C. Choi, "Iterative Methods for Singular Linear Equations
           and Least-Squares Problems", Dissertation,
           http://www.stanford.edu/group/SOL/dissertations/sou-cheng-choi-thesis.pdf
    """
    r = 0
    s = 0
    c = 0
    if b == 0:
        #return np.sign(a), 0, abs(a)
        c = tf.dtypes.cast(tf.math.sign(a), tf.float64)
        s = tf.Variable(0, dtype=tf.float64)
        r = tf.dtypes.cast(abs(a), tf.float64)
        #return tf.math.sign(a), 0, abs(a) 
    elif a == 0:
        #return 0, np.sign(b), abs(b)
        c = tf.Variable(0, dtype=tf.float64)
        s = tf.math.sign(b)
        r = abs(b)
        #return 0, tf.math.sign(b), abs(b) 
    elif abs(b) > abs(a):
        tau = a / b
        #s = np.sign(b) / sqrt(1 + tau * tau)
        s = tf.math.sign(b) / tf.sqrt(1 + tau * tau)
        c = s * tau
        r = b / s
    else:
        tau = b / a
        #c = np.sign(a) / sqrt(1+tau*tau)
        c = tf.dtypes.cast(tf.math.sign(a), tf.float64) / tf.sqrt(1+tau*tau)
        s = c * tau
        r = a / c
    return c, s, r

def matmat(A, X):
    #X = np.asanyarray(X)
    #if tf.rank(X) != 2:
    #    print("X shape is not correct", X.shape)
        #raise ValueError('expected 2-d ndarray or matrix, not', tf.rank(X))

    #if X.shape[0] != A.shape[1]:
    #    raise ValueError('dimension mismatch: %r, %r'
    #                  % (A.shape, X.shape))

    #Y = self._matmat(X)
    #Y = A.dot(X)
    Y = tf.tensordot(A, X, axes=1)

    #if isinstance(Y, np.matrix):
    #    Y = asmatrix(Y)

    return Y

def matvec(A, x):
    M,N = A.shape

    #x = np.asanyarray(x)

    #if x.shape != (N,) and x.shape != (N,1):
    #    raise ValueError('dimension mismatch: %r, %r'
    #                  % (A.shape, x.shape))

    #y = self._matvec(x)
    #y = A.matmat(x.reshape(-1, 1))
    #y = matmat(A, x.reshape(-1, 1))
    y = matmat(A, tf.reshape(x, [-1, 1]))

    #if isinstance(x, np.matrix):
    #    y = asmatrix(y)
    #else:
    #    y = np.asarray(y)

    #if tf.rank(x) == 1:
    #    #y = y.reshape(M)
    #    y = tf.reshape(y, (M,))
    #elif tf.rank(x) == 2:
    #    #y = y.reshape(M,1)
    #    y = tf.reshape(y, (M,1))
    #else:
    #    #raise ValueError('invalid shape returned by user-defined matvec()')
    #    #print('invalid shape returned by user-defined matvec()')
    #    print("shape:", y.shape)
    yt = tf.reshape(y, (M,))

    return yt

def rmatvec(A, x):
    m, n = A.shape
    x1 = x[:m]
    x2 = x[m:]

    #x = np.asanyarray(x)

    M,N = A.shape

    #if x.shape != (M,) and x.shape != (M,1):
    #    raise ValueError('dimension mismatch: %r, %r'
    #                  % (A.shape, x.shape))

    #y = self._rmatvec(x)
    #y = matvec(A.H, x)
    y = matvec(tf.linalg.adjoint(A), x)

    #if isinstance(x, np.matrix):
    #    y = asmatrix(y)
    #else:
    #    y = np.asarray(y)

    if tf.rank(x)  == 1:
        #y = y.reshape(N)
        y = tf.reshape(y, (N,))
    elif tf.rank(x)  == 2:
        #y = y.reshape(N,1)
        y = tf.reshape(y, (N,1))
    else:
        #raise ValueError('invalid shape returned by user-defined rmatvec()')
        print('invalid shape returned by user-defined rmatvec()')

    return y

