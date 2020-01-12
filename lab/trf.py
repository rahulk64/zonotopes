import numpy as np
import tensorflow as tf
import sys

from numpy.linalg import norm

from common import (
    EPS, step_size_to_bound, find_active_constraints, in_bounds,
    make_strictly_feasible, build_quadratic_1d, evaluate_quadratic,
    minimize_quadratic_1d, CL_scaling_vector, reflective_transformation,
    compute_grad)


@tf.function
def trf_linear(A, b, x_lsq, lb, ub, tol, lsq_solver, lsmr_tol, max_iter,
               verbose):
    b = tf.reshape(b, [tf.size(b)])
    lb = tf.reshape(lb, tf.shape(x_lsq))
    ub = tf.reshape(ub, tf.shape(x_lsq))

    tol = tf.constant(tol, dtype=tf.float64)
    m, n = A.shape
    x = reflective_transformation(x_lsq, lb, ub)
    x = tf.reshape(x, [tf.size(x)])
    x = make_strictly_feasible(x, lb, ub, rstep=0.1)
    tf.print("X:", x, output_stream=sys.stdout)

    r_aug = tf.zeros(m+n, dtype=tf.float64)
    auto_lsmr_tol = False
    if lsmr_tol is None:
        lsmr_tol = 1e-2 * tol
    elif lsmr_tol == 'auto':
        auto_lsmr_tol = True

    r = tf.tensordot(A, x, 1) - b
    g = compute_grad(A, r) 
    cost = 0.5 * tf.tensordot(r, r, 1)
    initial_cost = cost

    termination_status = 0 
    step_norm = None
    cost_change = None

    if max_iter is None:
        max_iter = 100

    for iteration in tf.range(max_iter):
        v, dv = CL_scaling_vector(x, g, lb, ub)
        tf.print("V:", v, output_stream=sys.stdout) 
        tf.print("DV:", dv, output_stream=sys.stdout) 
        g_scaled = tf.transpose(g) * v 
        g_norm = tf.norm(g_scaled, ord=np.inf)
        if g_norm < tol:
            termination_status = 1
        #else:
            #termination_status = 0

        if not termination_status == 0: 
            break

        #diag_h = np.diag(g.T * dv)
        #stuff = tf.transpose(g) * dv
        #stuff = tf.tensordot(g, dv, axes=1)
        stuff = g * tf.reshape(dv, [tf.size(dv), 1])
        diag_h = tf.linalg.diag_part(stuff)
        diag_root_h = tf.dtypes.cast(diag_h ** 0.5, dtype=tf.float64)
        d = v ** 0.5
        tf.print("D:", d, output_stream=sys.stdout)
        #g_h = d * np.squeeze(np.asarray(g.T))
        g_h = d * g

        #A_h = right_multiplied_operator(A, d)
        #lsmr_op = regularized_lsq_operator(A_h, diag_root_h)

        #r_aug[:m] = r
        #r_aug = tf.Variable(tf.convert_to_tensor(r_aug), dtype=tf.float64)
        #r_aug = tf.Variable(r_aug, dtype=tf.float64)
        #r_aug.assign(r_aug[:m].assign(r*tf.ones(m, dtype=tf.float64)))
        r_aug = tf.concat([tf.reshape(r, [tf.shape(r)[0]]), tf.zeros(n, dtype=tf.float64)], axis=0)

        if auto_lsmr_tol:
            eta = 1e-2 * min(0.5, g_norm)
            lsmr_tol = max(EPS, min(0.1, eta * g_norm))
        #p_h = -lsmr(lsmr_op, r_aug, atol=lsmr_tol, btol=lsmr_tol)[0]
        #p_h = -lsmr(A, r_aug, dis=d, diag=diag_root_h, atol=lsmr_tol, btol=lsmr_tol)[0]

        #A_h = A * np.diag(d)
        #temp = tf.reshape(tf.linalg.diag(d), [tf.shape(d)[0]])
        temp = tf.linalg.diag(d)
        #A_h = tf.Variable(A * temp, dtype=tf.float64)
        #A_h = tf.dtypes.cast(A * temp, dtype=tf.float64)
        A_h = tf.dtypes.cast(tf.matmul(A, temp), dtype=tf.float64)

        #lsmr_op = np.vstack(A_h, np.diag(diag_root_h))
        lsmr_op = tf.concat([A_h, tf.linalg.diag(diag_root_h)], axis=0)

        r_augl = tf.reshape(r_aug, (tf.size(r_aug), 1))
        p_h = -1 * tf.linalg.lstsq(lsmr_op, r_augl)
        p_h = tf.reshape(p_h, [tf.size(p_h)])

        tf.print("LSTSQ", p_h, output_stream=sys.stdout)

        p = d * p_h

        #p_dot_g = np.dot(p, g.T)
        p_dot_g = tf.tensordot(p, tf.transpose(g), 1) 

        if p_dot_g > 0:
            termination_status = -1

        theta = 1 - tf.math.minimum(tf.constant(0.005, dtype=tf.float64), g_norm)

        step = select_step(x, A, g_h, diag_h, p, p_h, d, lb, ub, theta, dis=d)
        cost_change = -evaluate_quadratic(A, g, step)

        # Perhaps almost never executed, the idea is that `p` is descent
        # direction thus we must find acceptable cost decrease using simple
        # "backtracking", otherwise the algorithm's logic would break.
        if cost_change < 0:
            x, step, cost_change = backtracking(
                A, g, x, p, theta, p_dot_g, lb, ub)
        else:
            x = make_strictly_feasible(x + step, lb, ub, rstep=0)

        step_norm = tf.norm(step)
        #r = A.dot(x) - b
        r = tf.tensordot(A, x, 1) - b
        #g = compute_grad(A, r.T).T
        g = compute_grad(A, tf.transpose(r))

        if cost_change < tol * cost:
            termination_status = 2

        #cost = 0.5 * np.dot(r, r.T)
        cost = 0.5 * tf.tensordot(r, r, 1)

    active_mask = find_active_constraints(x, lb, ub, rtol=tol)

    #return x_lsq
    return x

def backtracking(A, g, x, p, theta, p_dot_g, lb, ub):
    """Find an appropriate step size using backtracking line search."""
    alpha = tf.dtypes.cast(1., dtype=tf.float64)
    #while True:
    #    x_new, _ = reflective_transformation(x + alpha * p, lb, ub)
    #    step = x_new - x
    #    cost_change = -evaluate_quadratic(A, g, step)
    #    if cost_change > -0.1 * alpha * p_dot_g:
    #        break
    #    alpha *= 0.5
    x_new = reflective_transformation(x + alpha * p, lb, ub)
    step = x_new - x
    cost_change = -evaluate_quadratic(A, g, step)
    while not cost_change > -0.1 * alpha * p_dot_g:
        alpha *= 0.5
        x_new = reflective_transformation(x + alpha * p, lb, ub)
        step = x_new - x
        cost_change = -evaluate_quadratic(A, g, step)

    active = find_active_constraints(x_new, lb, ub)
    #if np.any(active != 0):
    if tf.reduce_any(active != 0):
        x_new = reflective_transformation(x + theta * alpha * p, lb, ub)
        x_new = make_strictly_feasible(x_new, lb, ub, rstep=0)
        step = x_new - x
        cost_change = -evaluate_quadratic(A, g, step)

    return x, step, cost_change


def select_step(x, A_h, g_h, c_h, p, p_h, d, lb, ub, theta, dis):
    """Select the best step according to Trust Region Reflective algorithm."""
    if in_bounds(x + p, lb, ub):
        return p

    p_stride, hits = step_size_to_bound(x, p, lb, ub)
    #r_h = np.copy(p_h)
    r_h = tf.identity(p_h)
    #r_h[hits.astype(bool)] *= -1
    #r_h[tf.dtypes.cast(hits, tf.bool)] *= -1
    r_h = tf.where(tf.equal(hits, False), r_h, r_h*-1)
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
        a, b, c = build_quadratic_1d(A_h, g_h, r_h, s0=p_h, diag=c_h, d=dis)
        r_stride, r_value = minimize_quadratic_1d(
            a, b, r_stride_l, r_stride_u, c=c)
        r_h = p_h + r_h * r_stride
        r = d * r_h
    else:
        r_value = tf.dtypes.cast(np.inf, tf.float64)

    # Now correct p_h to make it strictly interior.
    p_h *= theta
    p *= theta
    p_value = evaluate_quadratic(A_h, g_h, p_h, diag=c_h, d=dis)

    ag_h = -g_h
    ag = d * ag_h
    ag_stride_u, _ = step_size_to_bound(x, ag, lb, ub)
    ag_stride_u *= theta
    a, b = build_quadratic_1d(A_h, g_h, ag_h, diag=c_h, d=dis)
    ag_stride, ag_value = minimize_quadratic_1d(a, b, 0, ag_stride_u)
    ag *= ag_stride

    if p_value < r_value and p_value < ag_value:
        return p
    elif r_value < p_value and r_value < ag_value:
        return r
    else:
        return ag
