import numpy as np
from lsmr import projZonotope

#Example usage of projZonotope
A = np.matrix("-4 0 2 3 ; -2 1 0 -1")
b = np.matrix("20 10")
k = np.matrix("30 10")

# assumes zonotope centered at origin
c = k - b
c_eval = np.squeeze(np.asarray(c))

n = A.shape[0]
m = A.shape[1]

eps = projZonotope(A, c_eval, n, m)
print("eps", eps)
print("Ae", np.matmul(A, eps))
