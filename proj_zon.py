import numpy as np
# from numpy.linalg import norm
# from scipy.sparse import issparse, csr_matrix
# from scipy.sparse.linalg import LinearOperator, lsmr
# from scipy.optimize import OptimizeResult
from scipy.optimize import lsq_linear


""" MY CODE:
Implementation of the Frank-Wolfe Algorithm for projecting
a point outside of a zonotope back into the zonotope. This
problem has the form:

Inputs:
    A: matrix representation of the zonotope centered at origin
    b: center of zonotope in R^n
    k: point to project into zonotope
    max_iter: maximum number of iterations of FW
    tol: delta tolerance to terminate loop
    callback: TODO
"""
def zon_proj(A, b, k, max_iter=100, tol=1e-64, callback=None):
    eps = np.zeros((A.shape[1], 1))
    iter = 0
    for t in range(max_iter):
        iter = iter + 1
        A_eval = A*eps + b - k
        #print(A_eval)
        A_eval = np.resize(A_eval, (2,))
        s_t = getSt(A, A_eval)
        nu_t = getNu(t)
        eps_prev = eps
        eps = ((1 - nu_t)*eps + nu_t*s_t).T
        print("eps", eps.shape)
        if norm(eps - eps_prev) <= tol:
            break

    print(iter)
    return eps

def getNu(time, flag=0):
    if flag == 0:
        return 2/(2+time)
    else:
        raise Exception

def getSt(A, b):
    ones = np.ones(A.shape[1], )
    neg_ones = -1 * np.ones(A.shape[1],)
    print("ones", ones)
    print("neg_ones", neg_ones)
    # print("Ast", Ast)
    s_t = lsq_linear(A, b, bounds=(neg_ones, ones), lsq_solver='lsmr', lsmr_tol=1e-13, verbose=0).x
    print("size", s_t.shape)
    return s_t

A = np.matrix("-4 0 2 3 ; -2 1 0 -1")
b = np.matrix("20 ; 10")
k = np.matrix("16 ; 7")

# A = np.matrix("1 0 ; 0 1")
# b = np.matrix("0 ; 0")
# k = np.matrix("2 ; 0")

A_eval = k - b
#A_eval = np.resize(A_eval, (2,))
A_eval = np.squeeze(np.asarray(A_eval))

eps = getSt(A, A_eval)
print("A", A)
print("eps", eps)
print("Ae", np.matmul(A, eps))
print("end", np.add(np.matmul(A, eps).T, b))
