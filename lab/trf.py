import numpy as np
#import tensorflow as tf
from scipy.linalg import qr, solve_triangular
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import csr_matrix, dia_matrix

from numpy.linalg import norm

from lsm import lsmr 

from common import (
    EPS, step_size_to_bound, find_active_constraints, in_bounds,
    make_strictly_feasible, build_quadratic_1d, evaluate_quadratic,
    minimize_quadratic_1d, CL_scaling_vector, reflective_transformation,
    print_header_linear, print_iteration_linear, compute_grad,
    regularized_lsq_operator, right_multiplied_operator)

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
    eps = trf_linear(A, b, x_lsq, neg_ones, ones, 1e-13, 'lsmr', 1e-13, 200, 0)

    #NOTE THIS WORKS
    #eps = lsq_linear(A_coo, b, bounds=(neg_ones, ones), lsq_solver='lsmr', lsmr_tol=1e-13, verbose=0).x
    #eps = lsq_linear(A, b, bounds=(neg_ones, ones), lsq_solver='exact', lsmr_tol=1e-13, verbose=0).x
    #eps = tf.py_function(calcLSQ, [A, b], (tf.float64,tf.float64))
    return eps

def trf_linear(A, b, x_lsq, lb, ub, tol, lsq_solver, lsmr_tol, max_iter,
               verbose):
    m, n = A.shape
    x, _ = reflective_transformation(x_lsq, lb, ub)
    x = make_strictly_feasible(x, lb, ub, rstep=0.1)

    if lsq_solver == 'exact':
        QT, R, perm = qr(A, mode='economic', pivoting=True)
        QT = QT.T

        if m < n:
            R = np.vstack((R, np.zeros((n - m, n))))

        QTr = np.zeros(n)
        k = min(m, n)
    elif lsq_solver == 'lsmr':
        r_aug = np.zeros(m + n)
        auto_lsmr_tol = False
        if lsmr_tol is None:
            lsmr_tol = 1e-2 * tol
        elif lsmr_tol == 'auto':
            auto_lsmr_tol = True

    r = A.dot(x) - b
    g = compute_grad(A, r.T).T
    print(g.shape)
    cost = 0.5 * np.dot(r, r.T)
    initial_cost = cost

    termination_status = None
    step_norm = None
    cost_change = None

    if max_iter is None:
        max_iter = 100

    if verbose == 2:
        print_header_linear()

    for iteration in range(max_iter):
        print("iteration")
        v, dv = CL_scaling_vector(x, g, lb, ub)
        g_scaled = g.T * v 
        g_norm = norm(g_scaled, ord=np.inf)
        if g_norm < tol:
            termination_status = 1

        if verbose == 2:
            print_iteration_linear(iteration, cost, cost_change,
                                   step_norm, g_norm)

        if termination_status is not None:
            break

        diag_h = np.diag(g.T * dv)
        diag_root_h = diag_h ** 0.5
        d = v ** 0.5
        g_h = d * np.squeeze(np.asarray(g.T))

        A_h = right_multiplied_operator(A, d)
        if lsq_solver == 'exact':
            QTr[:k] = QT.dot(r)
            p_h = -regularized_lsq_with_qr(m, n, R * d[perm], QTr, perm,
                                           diag_root_h, copy_R=False)
        elif lsq_solver == 'lsmr':
            lsmr_op = regularized_lsq_operator(A_h, diag_root_h)
            r_aug[:m] = r
            if auto_lsmr_tol:
                eta = 1e-2 * min(0.5, g_norm)
                lsmr_tol = max(EPS, min(0.1, eta * g_norm))
            p_h = -lsmr(lsmr_op, r_aug, atol=lsmr_tol, btol=lsmr_tol)[0]

        p = d * p_h

        p_dot_g = np.dot(p, g.T)
        if p_dot_g > 0:
            termination_status = -1

        theta = 1 - min(0.005, g_norm)

        step = select_step(x, A_h, g_h, diag_h, p, p_h, d, lb, ub, theta)
        cost_change = -evaluate_quadratic(A, g, step)

        # Perhaps almost never executed, the idea is that `p` is descent
        # direction thus we must find acceptable cost decrease using simple
        # "backtracking", otherwise the algorithm's logic would break.
        if cost_change < 0:
            x, step, cost_change = backtracking(
                A, g, x, p, theta, p_dot_g, lb, ub)
        else:
            x = make_strictly_feasible(x + step, lb, ub, rstep=0)

        step_norm = norm(step)
        r = A.dot(x) - b
        g = compute_grad(A, r.T).T

        if cost_change < tol * cost:
            termination_status = 2

        cost = 0.5 * np.dot(r, r.T)

    if termination_status is None:
        termination_status = 0

    active_mask = find_active_constraints(x, lb, ub, rtol=tol)

    return x

def backtracking(A, g, x, p, theta, p_dot_g, lb, ub):
    """Find an appropriate step size using backtracking line search."""
    alpha = 1
    while True:
        x_new, _ = reflective_transformation(x + alpha * p, lb, ub)
        step = x_new - x
        cost_change = -evaluate_quadratic(A, g, step)
        if cost_change > -0.1 * alpha * p_dot_g:
            break
        alpha *= 0.5

    active = find_active_constraints(x_new, lb, ub)
    if np.any(active != 0):
        x_new, _ = reflective_transformation(x + theta * alpha * p, lb, ub)
        x_new = make_strictly_feasible(x_new, lb, ub, rstep=0)
        step = x_new - x
        cost_change = -evaluate_quadratic(A, g, step)

    return x, step, cost_change


def select_step(x, A_h, g_h, c_h, p, p_h, d, lb, ub, theta):
    """Select the best step according to Trust Region Reflective algorithm."""
    if in_bounds(x + p, lb, ub):
        return p

    p_stride, hits = step_size_to_bound(x, p, lb, ub)
    r_h = np.copy(p_h)
    r_h[hits.astype(bool)] *= -1
    r = d * r_h

    # Restrict step, such that it hits the bound.
    p *= p_stride
    p_h *= p_stride
    x_on_bound = x + p

    # Find the step size along reflected direction.
    r_stride_u, _ = step_size_to_bound(x_on_bound, r, lb, ub)

    # Stay interior.
    r_stride_l = (1 - theta) * r_stride_u
    r_stride_u *= theta

    if r_stride_u > 0:
        a, b, c = build_quadratic_1d(A_h, g_h, r_h, s0=p_h, diag=c_h)
        r_stride, r_value = minimize_quadratic_1d(
            a, b, r_stride_l, r_stride_u, c=c)
        r_h = p_h + r_h * r_stride
        r = d * r_h
    else:
        r_value = np.inf

    # Now correct p_h to make it strictly interior.
    p_h *= theta
    p *= theta
    p_value = evaluate_quadratic(A_h, g_h, p_h, diag=c_h)

    ag_h = -g_h
    ag = d * ag_h
    ag_stride_u, _ = step_size_to_bound(x, ag, lb, ub)
    ag_stride_u *= theta
    a, b = build_quadratic_1d(A_h, g_h, ag_h, diag=c_h)
    ag_stride, ag_value = minimize_quadratic_1d(a, b, 0, ag_stride_u)
    ag *= ag_stride

    if p_value < r_value and p_value < ag_value:
        return p
    elif r_value < p_value and r_value < ag_value:
        return r
    else:
        return ag

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
