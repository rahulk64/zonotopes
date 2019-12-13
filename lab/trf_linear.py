import numpy as np
from linop import LinearOperator

from numpy.linalg import norm

from lsm import lsmr 

from common import (
    EPS, step_size_to_bound, find_active_constraints, in_bounds,
    make_strictly_feasible, build_quadratic_1d, evaluate_quadratic,
    minimize_quadratic_1d, CL_scaling_vector, reflective_transformation,
    print_header_linear, print_iteration_linear, compute_grad,
    regularized_lsq_operator, right_multiplied_operator)
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
    g = compute_grad(A, r)
    cost = 0.5 * np.dot(r, r)
    initial_cost = cost

    termination_status = None
    step_norm = None
    cost_change = None

    if max_iter is None:
        max_iter = 100

    if verbose == 2:
        print_header_linear()

    for iteration in range(max_iter):
        v, dv = CL_scaling_vector(x, g, lb, ub)
        g_scaled = g * v
        g_norm = norm(g_scaled, ord=np.inf)
        if g_norm < tol:
            termination_status = 1

        if verbose == 2:
            print_iteration_linear(iteration, cost, cost_change,
                                   step_norm, g_norm)

        if termination_status is not None:
            break

        diag_h = g * dv
        diag_root_h = diag_h ** 0.5
        d = v ** 0.5
        g_h = d * g

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

        p_dot_g = np.dot(p, g)
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
        g = compute_grad(A, r)

        if cost_change < tol * cost:
            termination_status = 2

        cost = 0.5 * np.dot(r, r)

    if termination_status is None:
        termination_status = 0

    active_mask = find_active_constraints(x, lb, ub, rtol=tol)

    return OptimizeResult(
        x=x, fun=r, cost=cost, optimality=g_norm, active_mask=active_mask,
        nit=iteration + 1, status=termination_status,
        initial_cost=initial_cost)
