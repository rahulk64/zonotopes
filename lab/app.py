from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from proj import projZonotope

# Example usage of projZonotope
# A is zonotope matrix, b is center of zonotope
A = np.matrix("-4 0 2 3 ; -2 1 0 -1")
b = np.matrix("20 10")

# k is value to project into zonotope
k = np.matrix("27. 16.")

# projection assumes zonotope centered at origin
A_eval = k - b

# need to squeeze for dimension alignment
A_eval = np.squeeze(np.asarray(A_eval))

A = tf.convert_to_tensor(A)
A_eval = tf.convert_to_tensor(A_eval)

print("A SHAPE", A.shape)
print("A_eval", A_eval.shape)

eps = projZonotope(A, A_eval)
with tf.Session() as sess:
    e = sess.run(eps)
    print("eps", eps)
print("eps", eps)
print("Ae", np.matmul(A, eps))
"""
with tf.Session() as sess:
    e = sess.run(eps)
    print("A", A)
    print("eps", eps)
    print("Ae", np.matmul(A, eps))
    print("end", np.add(np.matmul(A, eps), b))
"""
