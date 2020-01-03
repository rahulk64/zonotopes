"""Functions used by least-squares algorithms."""
from __future__ import division, print_function, absolute_import
from tensorflow.python.ops import bitwise_ops

from math import copysign

import numpy as np
from numpy.linalg import norm
import tensorflow as tf

from linop import LinearOperator, aslinearoperator

EPS = np.finfo(float).eps

# Functions related to a trust-region problem.


def build_quadratic_1d(J, g, s, diag=None, s0=None):
    v = J.dot(s)
    a = np.dot(v, v)
    if diag is not None:
        a += np.dot(s * diag, s)
    a *= 0.5

    b = np.dot(g, s)

    if s0 is not None:
        u = J.dot(s0)
        b += np.dot(u, v)
        c = 0.5 * np.dot(u, u) + np.dot(g, s0)
        if diag is not None:
            b += np.dot(s0 * diag, s)
            c += 0.5 * np.dot(s0 * diag, s0)
        return a, b, c
    else:
        return a, b


def minimize_quadratic_1d(a, b, lb, ub, c=0):
    t = [lb, ub]
    if a != 0:
        extremum = -0.5 * b / a
        if lb < extremum < ub:
            t.append(extremum)
    t = np.asarray(t)
    y = t * (a * t + b) + c
    min_index = np.argmin(y)
    return t[min_index], y[min_index]


def evaluate_quadratic(J, g, s, diag=None):
    if s.ndim == 1:
        Js = J.dot(s)
        q = np.vdot(Js, Js)
        if diag is not None:
            q += np.dot(s * diag, s)
    else:
        Js = J.dot(s.T)
        q = np.sum(Js**2, axis=0)
        if diag is not None:
            q += np.sum(diag * s**2, axis=1)

    l = np.dot(s, g.T)

    return 0.5 * q + l


# Utility functions to work with bound constraints.

@tf.function
def in_bounds(x, lb, ub):
    return tf.reduce_all((x >= lb) & (x <= ub))


def step_size_to_bound(x, s, lb, ub):
    non_zero = np.nonzero(s)
    s_non_zero = s[non_zero]
    steps = np.empty_like(x)
    steps.fill(np.inf)
    with np.errstate(over='ignore'):
        steps[non_zero] = np.maximum((lb - x)[non_zero] / s_non_zero,
                                     (ub - x)[non_zero] / s_non_zero)
    min_step = np.min(steps)
    return min_step, np.equal(steps, min_step) * np.sign(s).astype(int)


def find_active_constraints(x, lb, ub, rtol=1e-10):
    active = tf.zeros_like(x, dtype=tf.int32)
    lb = tf.reshape(lb, tf.shape(x)) 
    ub = tf.reshape(ub, tf.shape(x))

    if rtol == 0:
        active[x <= lb] = -1
        active[x >= ub] = 1
        return active

    lower_dist = x - lb
    upper_dist = ub - x

    one = tf.Variable(1.0, dtype=tf.float64)

    lower_threshold = rtol * tf.maximum(one, tf.abs(lb))
    upper_threshold = rtol * tf.maximum(one, tf.abs(ub))

    lower_active = (tf.math.is_finite(lb) &
                    (lower_dist <= tf.minimum(upper_dist, lower_threshold)))
    #active[lower_active] = -1
    #indices = tf.dtypes.cast(lower_active, tf.int32)
    updates = tf.dtypes.cast(-1*tf.ones_like(active), tf.int32)
    #active = tf.tensor_scatter_nd_update(active, indices, updates)
    active = tf.where(tf.equal(lower_active,False), active, updates)

    upper_active = (tf.math.is_finite(ub) &
                    (upper_dist <= tf.minimum(lower_dist, upper_threshold)))
    #active[upper_active] = 1
    #indices2 = tf.dtypes.cast(upper_active, tf.int32)
    updates2 = tf.dtypes.cast(tf.ones_like(active), tf.int32)
    #active = tf.tensor_scatter_nd_update(active, indices2, updates2)
    active = tf.where(tf.equal(upper_active,False), active, updates2)

    return active


def make_strictly_feasible(x, lb, ub, rstep=1e-10):
    x_new = tf.identity(x) #x.copy()
    lb = tf.reshape(lb, tf.shape(x))
    ub = tf.reshape(ub, tf.shape(x))

    active = find_active_constraints(x, lb, ub, rstep)
    lower_mask = tf.equal(active, -1)
    upper_mask = tf.equal(active, 1)

    if rstep == 0:
        x_new[lower_mask] = tf.nextafter(lb[lower_mask], ub[lower_mask])
        x_new[upper_mask] = tf.nextafter(ub[upper_mask], lb[upper_mask])
    else:
        #x_new[lower_mask] = (lb[lower_mask] +
        #                     rstep * tf.maximum(tf.constant(1.0, dtype=tf.float64), tf.abs(lb[lower_mask])))
        x_new = tf.where(tf.equal(lower_mask, False), x_new, lb[lower_mask] +
                             rstep * tf.maximum(tf.constant(1.0, dtype=tf.float64), tf.abs(lb[lower_mask])))

        #x_new[upper_mask] = (ub[upper_mask] -
        #                     rstep * tf.maximum(tf.constant(1.0, dtype=tf.float64), tf.abs(ub[upper_mask])))
        x_new = tf.where(tf.equal(upper_mask, False), x_new, ub[upper_mask] -
                             rstep * tf.maximum(tf.constant(1.0, dtype=tf.float64), tf.abs(ub[upper_mask])))
        
    tight_bounds = (x_new < lb) | (x_new > ub)
    #x_new[tight_bounds] = 0.5 * (lb[tight_bounds] + ub[tight_bounds])
    x_new = tf.where(tf.equal(tight_bounds, False), x_new, 0.5 * (lb[tight_bounds] + ub[tight_bounds])) 

    return x_new


def CL_scaling_vector(x, g, lb, ub):
    #v = np.ones_like(x)
    #dv = np.zeros_like(x)
    v = tf.ones_like(x, dtype=tf.float64)
    dv = tf.zeros_like(x, dtype=tf.float64)

    #mask = ((g < 0) & np.isfinite(ub))
    mask = ((g < 0) & tf.math.is_finite(ub))
    #mask = np.squeeze(np.asarray(mask))

    v[mask] = ub[mask] - x[mask]
    dv[mask] = -1

    mask = (g > 0) & np.isfinite(lb)
    mask = np.squeeze(np.asarray(mask))
    v[mask] = x[mask] - lb[mask]
    dv[mask] = 1

    v.reshape(x.shape)
    dv.reshape(x.shape)

    return v, dv

def reflective_transformation(y, lb, ub):
    if in_bounds(y, lb, ub):
        return y, tf.ones_like(y)

    lb_finite = tf.math.is_finite(lb)
    ub_finite = tf.math.is_finite(ub)

    x = tf.dtypes.cast(tf.identity(y), dtype=tf.float64)
    g_negative = tf.zeros_like(y, dtype=bool)

    mask = lb_finite & ~ub_finite
    #x[mask] = tf.maximum(y[mask], 2 * lb[mask] - y[mask])
    #g_negative[mask] = y[mask] < lb[mask]
    x = tf.where(tf.equal(mask, False), x, tf.maximum(y[mask], 2 * lb[mask] - y[mask]))
    g_negative = tf.where(tf.equal(mask, False), g_negative, y[mask] < lb[mask])

    mask = ~lb_finite & ub_finite
    #x[mask] = np.minimum(y[mask], 2 * ub[mask] - y[mask])
    #g_negative[mask] = y[mask] > ub[mask]
    x = tf.where(tf.equal(mask, False), x, tf.minimum(y[mask], 2 * ub[mask] - y[mask]))
    g_negative = tf.where(tf.equal(mask, False), g_negative, y[mask] > ub[mask])

    mask = lb_finite & ub_finite
    d = ub - lb
    tmp = y[mask] - lb[mask]
    t = tf.math.floormod(y[mask] - lb[mask], 2 * d[mask])
    #x[mask] = lb[mask] + np.minimum(t, 2 * d[mask] - t)
    #g_negative[mask] = t > d[mask]
    x = tf.where(tf.equal(mask, False), x, lb[mask] + tf.minimum(t, 2 * d[mask] - t))
    g_negative = tf.where(tf.equal(mask, False), g_negative, t > d[mask])

    g = tf.ones_like(y)
    #g[g_negative] = -1
    g = tf.where(tf.equal(g_negative, False), g, -1)

    return x, g


# Functions to display algorithm's progress.

def print_header_linear():
    print("{0:^15}{1:^15}{2:^15}{3:^15}{4:^15}"
          .format("Iteration", "Cost", "Cost reduction", "Step norm",
                  "Optimality"))


def print_iteration_linear(iteration, cost, cost_reduction, step_norm,
                           optimality):
    if cost_reduction is None:
        cost_reduction = " " * 15
    else:
        cost_reduction = "{0:^15.2e}".format(cost_reduction)

    if step_norm is None:
        step_norm = " " * 15
    else:
        step_norm = "{0:^15.2e}".format(step_norm)

    print("{0:^15}{1:^15.4e}{2}{3}{4:^15.2e}".format(
        iteration, cost, cost_reduction, step_norm, optimality))


# Simple helper functions.


def compute_grad(J, f):
    if isinstance(J, LinearOperator):
        return J.rmatvec(f)
    else:
        #return J.T.dot(f)
        return tf.tensordot(tf.transpose(J), f, 1)


def right_multiplied_operator(J, d):
    J = aslinearoperator(J)

    def matvec(x):
        return J.matvec(np.ravel(x) * d)

    def matmat(X):
        return J.matmat(X * d[:, np.newaxis])

    def rmatvec(x):
        return d * J.rmatvec(x)

    return LinearOperator(J.shape, matvec=matvec, matmat=matmat,
                          rmatvec=rmatvec)


def regularized_lsq_operator(J, diag):
    J = aslinearoperator(J)
    m, n = J.shape

    def matvec(x):
        return np.hstack((J.matvec(x), diag * x))

    def rmatvec(x):
        x1 = x[:m]
        x2 = x[m:]
        return J.rmatvec(x1) + diag * x2

    return LinearOperator((m + n, n), matvec=matvec, rmatvec=rmatvec)

