"""Functions used by least-squares algorithms."""
from __future__ import division, print_function, absolute_import
from tensorflow.python.ops import bitwise_ops

from math import copysign

import numpy as np
from numpy.linalg import norm
import tensorflow as tf

from ops import matvec

EPS = np.finfo(float).eps

# Functions related to a trust-region problem.
def JDot(J, x, d):
    d = tf.reshape(d, [d.shape[0]])
    temp = tf.reshape(x, [-1]) * d
    val = matvec(J, temp)
    return matvec(J, tf.reshape(x, [-1]) * d)

def build_quadratic_1d(J, g, s, diag=None, s0=None, d=None):
    v = JDot(J, s, d)
    s = tf.reshape(s, [-1])
    g = tf.reshape(g, [-1])
    a = tf.tensordot(v, v, 1)
    if diag is not None:
        a += tf.tensordot(s * diag, s, 1)
    a *= 0.5

    b = tf.tensordot(g, s, 1)

    if s0 is not None:
        s0 = tf.reshape(s0, [-1])
        u = JDot(J, s0, d)
        b += tf.tensordot(u, v, 1)
        c = 0.5 * tf.tensordot(u, u, 1) + tf.tensordot(g, s0, 1)
        if diag is not None:
            b += tf.tensordot(s0 * diag, s, 1)
            c += 0.5 * tf.tensordot(s0 * diag, s0, 1)
        return a, b, c
    else:
        return a, b


def minimize_quadratic_1d(a, b, lb, ub, c=0):
    ly = lb * (a * lb + b) + c
    uy = ub * (a * ub + b) + c
    if a != 0:
        extremum = -0.5 * b / a
        first = lb < extremum
        second = extremum < ub
        if first and second:
            mid = extremum * (a * extremum + b) + c
            if ly < uy and ly < mid:
                return tf.dtypes.cast(lb, tf.float64), ly
            elif uy < ly and uy < mid:
                return tf.dtypes.cast(ub, tf.float64), uy
            else:
                return extremum, mid
        else:
            if ly < uy:
                return tf.dtypes.cast(lb, tf.float64), ly
            else:
                return tf.dtypes.cast(ub, tf.float64), uy
    else:
        if ly < uy:
            return tf.dtypes.cast(lb, tf.float64), ly
        else:
            return tf.dtypes.cast(ub, tf.float64), uy


def evaluate_quadratic(J, g, s, diag=None, d=None):
    if d is not None:
        Js = JDot(J, s, d)
    else:
        Js = tf.tensordot(J, s, 1)
    q = tf.tensordot(Js, Js, 1)
    if diag is not None:
        q += tf.tensordot(s * diag, s, 1)

    l = tf.tensordot(s, tf.transpose(g), 1)

    return 0.5 * q + l


# Utility functions to work with bound constraints.

@tf.function
def in_bounds(x, lb, ub):
    return tf.reduce_all((x >= lb) & (x <= ub))


def step_size_to_bound(x, s, lb, ub):
    lb = tf.reshape(lb, [tf.size(lb)])
    ub = tf.reshape(ub, [tf.size(ub)])
    zero = tf.constant(0, dtype=tf.float64)
    non_zero = tf.not_equal(s, zero)

    s_non_zero = s[non_zero]
    steps = tf.dtypes.cast(tf.fill(tf.shape(x), np.inf), dtype=tf.float64)
    with np.errstate(over='ignore'):
        steps = tf.where(tf.equal(non_zero, False), steps, tf.math.maximum((lb - x) / s_non_zero, (ub - x) / s_non_zero))
    min_step = tf.math.reduce_min(steps)
    return min_step, tf.dtypes.cast(tf.equal(steps, min_step), tf.int32) * tf.dtypes.cast(tf.sign(s), tf.int32)


def find_active_constraints(x, lb, ub, rtol=1e-10):
    active = tf.zeros_like(x, dtype=tf.int32)
    lb = tf.reshape(lb, tf.shape(x)) 
    ub = tf.reshape(ub, tf.shape(x))

    if rtol == 0:
        active = tf.where(tf.equal(x <= lb, False), active, -1)
        active = tf.where(tf.equal(x >= ub, False), active, 1)
        return active

    lower_dist = x - lb
    upper_dist = ub - x

    one = tf.ones_like(lb, dtype=tf.float64)

    lower_threshold = rtol * tf.maximum(one, tf.abs(lb))
    upper_threshold = rtol * tf.maximum(one, tf.abs(ub))

    lower_active = (tf.math.is_finite(lb) &
                    (lower_dist <= tf.minimum(upper_dist, lower_threshold)))
    updates = tf.dtypes.cast(-1*tf.ones_like(active), tf.int32)
    active = tf.where(tf.equal(lower_active,False), active, updates)

    upper_active = (tf.math.is_finite(ub) &
                    (upper_dist <= tf.minimum(lower_dist, upper_threshold)))
    updates2 = tf.dtypes.cast(tf.ones_like(active), tf.int32)
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
        x_new = tf.where(tf.equal(lower_mask, False), x_new, tf.math.nextafter(lb, ub))
        x_new = tf.where(tf.equal(upper_mask, False), x_new, tf.math.nextafter(ub, lb))
    else:
        x_new = tf.where(tf.equal(lower_mask, False), x_new, lb +
                             rstep * tf.maximum(tf.constant(1.0, dtype=tf.float64), tf.abs(lb)))

        x_new = tf.where(tf.equal(upper_mask, False), x_new, ub -
                             rstep * tf.maximum(tf.constant(1.0, dtype=tf.float64), tf.abs(ub)))
        
    tight_bounds = (x_new < lb) | (x_new > ub)
    x_new = tf.where(tf.equal(tight_bounds, False), x_new, 0.5 * (lb + ub)) 

    return x_new


def CL_scaling_vector(x, g, lb, ub):
    lb = tf.reshape(lb, tf.shape(x))
    ub = tf.reshape(ub, tf.shape(x))
    v = tf.ones_like(x, dtype=tf.float64)
    dv = tf.zeros_like(x, dtype=tf.float64)

    mask = ((g < 0) & tf.math.is_finite(ub))

    v = tf.where(tf.equal(mask, False), v, ub-x)
    dv = tf.where(tf.equal(mask, False), dv, -1)

    mask = ((g > 0) & tf.math.is_finite(lb))
    v = tf.where(tf.equal(mask, False), v, x-lb)
    dv = tf.where(tf.equal(mask, False), dv, 1)

    v = tf.reshape(v, tf.shape(x))
    dv = tf.reshape(dv, tf.shape(x))

    return v, dv

def reflective_transformation(y, lb, ub):
    if in_bounds(y, lb, ub):
        return y

    y = tf.reshape(y, [tf.size(y)])
    lb = tf.reshape(lb, [tf.size(lb)])
    ub = tf.reshape(ub, [tf.size(ub)])

    lb_finite = tf.math.is_finite(lb)
    ub_finite = tf.math.is_finite(ub)

    x = tf.dtypes.cast(tf.identity(y), dtype=tf.float64)

    mask = lb_finite & ~ub_finite
    x = tf.where(tf.equal(mask, False), x, tf.maximum(y, 2 * lb - y))

    mask = ~lb_finite & ub_finite
    x = tf.where(tf.equal(mask, False), x, tf.minimum(y, 2 * ub - y))

    mask = lb_finite & ub_finite
    d = ub - lb
    tmp = y[mask] - lb[mask]
    t = tf.math.floormod(y[mask] - lb[mask], 2 * d[mask])
    x = tf.where(tf.equal(mask, False), x, lb + tf.minimum(t, 2 * d - t))

    return x


def compute_grad(J, f):
    return tf.tensordot(tf.transpose(J), f, 1)

