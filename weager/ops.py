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
