import numpy as np
from proj import projZonotope

# example usage of projZonotope
# A is zonotope matrix, b is center of zonotope
A = np.matrix("-4 0 2 3 ; -2 1 0 -1")
b = np.matrix("20 10")

# k is value to project into zonotope
k = np.matrix("27. 16.") 

# projection assumes zonotope centered at origin
c = k - b

# need to squeeze for dimension alignment 
c_eval = np.squeeze(np.asarray(c))

eps = projZonotope(A, c_eval) 

# prints values for projection
print("eps", eps)
print("Ae", np.matmul(A, eps))

print("projecting from: ", k)
print("projection point: ", np.matmul(A, eps) + b) # should receive (26, 13)
