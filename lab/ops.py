import numpy as np
import sys
import tensorflow as tf

from numpy import zeros, infty, atleast_1d, result_type
from numpy.linalg import norm
from math import sqrt

def matmat(A, X):
    Y = tf.tensordot(A, X, axes=1)
    return Y

def matvec(A, x):
    M,N = A.shape
    y = matmat(A, tf.reshape(x, [-1, 1]))
    yt = tf.reshape(y, (M,))

    return yt

def rmatvec(A, x):
    m, n = A.shape
    x1 = x[:m]
    x2 = x[m:]

    M,N = A.shape

    y = matvec(tf.linalg.adjoint(A), x)

    if tf.rank(x)  == 1:
        y = tf.reshape(y, (N,))
    elif tf.rank(x)  == 2:
        y = tf.reshape(y, (N,1))
    else:
        print('invalid shape returned by user-defined rmatvec()')

    return y

