import numpy as np
#import tensorflow as tf
from scipy.optimize import lsq_linear
from scipy.sparse import csr_matrix, dia_matrix

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
def projZonotope(A_arg, b, n, m):
    #un-ravel
    #A = tf.reshape(A_arg, (n,m))
    #A = A_arg
    ones = np.squeeze(np.asarray(np.ones(A.shape[1], )))
    neg_ones = -1 * ones
    #A_coo = csr_matrix(A)
    x_lsq = np.linalg.lstsq(A, b, rcond=-1)[0]
    #eps = bvls(A, b, x_lsq, neg_ones, ones, 1e-13, 200, 2)

    #NOTE THIS WORKS
    #eps = lsq_linear(A_coo, b, bounds=(neg_ones, ones), lsq_solver='lsmr', lsmr_tol=1e-13, verbose=0).x
    eps = lsq_linear(A, b, bounds=(neg_ones, ones), lsq_solver='exact', lsmr_tol=1e-13, verbose=0).x
    #eps = tf.py_function(calcLSQ, [A, b], (tf.float64,tf.float64))
    return eps
"""
def calcLSQ(A, b):
    A_coo = csr_matrix(A)
    eps = lsq_linear(A_coo, b, bounds=(neg_ones, ones), lsq_solver='lsmr', lsmr_tol=1e-13, verbose=0).x
"""

import numpy as np
from numpy.linalg import norm, lstsq

#@tf.function
def compute_kkt_optimality(g, on_bound):
    """Compute the maximum violation of KKT conditions."""
    g_kkt = g * on_bound
    free_set = on_bound == 0
    g_kkt[free_set] = np.abs(g[free_set])
    return np.max(g_kkt)

#@tf.function
def bvls(A, b, x_lsq, lb, ub, tol, max_iter, verbose):
    m, n = A.shape

    x = x_lsq.copy()
    on_bound = np.zeros(n)

    mask = x < lb
    x[mask] = lb[mask]
    on_bound[mask] = -1

    mask = x > ub
    x[mask] = ub[mask]
    on_bound[mask] = 1

    free_set = on_bound == 0
    active_set = ~free_set
    free_set, = np.nonzero(free_set)

    r = A.dot(x) - b
    cost = 0.5 * np.dot(r.T, r)
    initial_cost = cost
    g = A.T.dot(r.T)

    cost_change = None
    step_norm = None
    iteration = 0


    # This is the initialization loop. The requirement is that the
    # least-squares solution on free variables is feasible before BVLS starts.
    # One possible initialization is to set all variables to lower or upper
    # bounds, but many iterations may be required from this state later on.
    # The implemented ad-hoc procedure which intuitively should give a better
    # initial state: find the least-squares solution on current free variables,
    # if its feasible then stop, otherwise set violating variables to
    # corresponding bounds and continue on the reduced set of free variables.

    while free_set.size > 0:
        iteration += 1
        x_free_old = x[free_set].copy()

        A_free = A[:, free_set]
        b_free = b - A.dot(x * active_set)
        z = lstsq(A_free, b_free, rcond=-1)[0]

        lbv = z < lb[free_set]
        ubv = z > ub[free_set]
        v = lbv | ubv

        if np.any(lbv):
            ind = free_set[lbv]
            x[ind] = lb[ind]
            active_set[ind] = True
            on_bound[ind] = -1

        if np.any(ubv):
            ind = free_set[ubv]
            x[ind] = ub[ind]
            active_set[ind] = True
            on_bound[ind] = 1

        ind = free_set[~v]
        x[ind] = z[~v]

        r = A.dot(x) - b
        cost_new = 0.5 * np.dot(r, r)
        cost_change = cost - cost_new
        cost = cost_new
        g = A.T.dot(r)
        step_norm = norm(x[free_set] - x_free_old)

        if np.any(v):
            free_set = free_set[~v]
        else:
            break

    if max_iter is None:
        max_iter = n
    max_iter += iteration

    termination_status = None

    # Main BVLS loop.

    optimality = compute_kkt_optimality(g, on_bound)
    for iteration in range(iteration, max_iter):
        if optimality < tol:
            termination_status = 1

        if termination_status is not None:
            break

        move_to_free = np.argmax(g * on_bound)
        on_bound[move_to_free] = 0
        free_set = on_bound == 0
        active_set = ~free_set
        free_set, = np.nonzero(free_set)

        x_free = x[free_set]
        x_free_old = x_free.copy()
        lb_free = lb[free_set]
        ub_free = ub[free_set]

        A_free = A[:, free_set]
        b_free = b - A.dot(x * active_set)
        z = lstsq(A_free, b_free, rcond=-1)[0]

        lbv, = np.nonzero(z < lb_free)
        ubv, = np.nonzero(z > ub_free)
        v = np.hstack((lbv, ubv))

        if v.size > 0:
            alphas = np.hstack((
                lb_free[lbv] - x_free[lbv],
                ub_free[ubv] - x_free[ubv])) / (z[v] - x_free[v])

            i = np.argmin(alphas)
            i_free = v[i]
            alpha = alphas[i]

            x_free *= 1 - alpha
            x_free += alpha * z

            if i < lbv.size:
                on_bound[free_set[i_free]] = -1
            else:
                on_bound[free_set[i_free]] = 1
        else:
            x_free = z

        x[free_set] = x_free
        step_norm = norm(x_free - x_free_old)

        r = A.dot(x) - b
        cost_new = 0.5 * np.dot(r, r)
        cost_change = cost - cost_new

        if cost_change < tol * cost:
            termination_status = 2
        cost = cost_new

        g = A.T.dot(r)
        optimality = compute_kkt_optimality(g, on_bound)

    if termination_status is None:
        termination_status = 0

    return x
"""    return OptimizeResult(
        x=x, fun=r, cost=cost, optimality=optimality, active_mask=on_bound,
        nit=iteration + 1, status=termination_status,
        initial_cost=initial_cost)
"""
#Example usage of projZonotope
#A = tf.constant(np.matrix("-4 0 2 3 ; -2 1 0 -1"))
A = np.matrix("-4 0 2 3 ; -2 1 0 -1")
b = np.matrix("20 10")
k = np.matrix("30 10")

#should probs include this in the function
A_eval = k - b
A_eval = np.squeeze(np.asarray(A_eval))

n = A.shape[0]
m = A.shape[1]

#A_arg = np.asarray(A).ravel()
#print(A_arg)
#A_arg = A

eps = projZonotope(A, A_eval, n, m)
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
