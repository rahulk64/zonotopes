"""Functions used by least-squares algorithms."""
from __future__ import division, print_function, absolute_import

from math import copysign

import numpy as np
from numpy.linalg import norm

from lsmr import matvec

EPS = np.finfo(float).eps

# Functions related to a trust-region problem.

def JDot(J, x, d):
    x = np.asarray(x)

    if x.ndim == 1 or x.ndim == 2 and x.shape[1] == 1:
        return matvec(J, np.ravel(x) * d)
    elif x.ndim == 2:
        print("oh no this is bad")
        return J.matmat(x)
    else:
        raise ValueError('expected 1-d or 2-d array or matrix, got %r'
                        % x)

def build_quadratic_1d(J, g, s, diag=None, s0=None, d=None):
    #v = J.dot(s)
    v = JDot(J, s, d)
    a = np.dot(v, v)
    if diag is not None:
        a += np.dot(s * diag, s)
    a *= 0.5

    b = np.dot(g, s)

    if s0 is not None:
        #u = J.dot(s0)
        u = JDot(J, s0, d)
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


def evaluate_quadratic(J, g, s, diag=None, d=None):
    if s.ndim == 1:
        #Js = J.dot(s)
        if d is not None:
            Js = JDot(J, s, d)
        else:
            Js = J.dot(s)
        q = np.vdot(Js, Js)
        if diag is not None:
            q += np.dot(s * diag, s)
    else:
        #Js = J.dot(s.T)
        if d is not None:
            Js = JDot(J, s.T, d)
        else:
            Js = J.dot(s.T)
        q = np.sum(Js**2, axis=0)
        if diag is not None:
            q += np.sum(diag * s**2, axis=1)

    l = np.dot(s, g.T)

    return 0.5 * q + l


# Utility functions to work with bound constraints.


def in_bounds(x, lb, ub):
    return np.all((x >= lb) & (x <= ub))


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
    active = np.zeros_like(x, dtype=int)

    if rtol == 0:
        active[x <= lb] = -1
        active[x >= ub] = 1
        return active

    lower_dist = x - lb
    upper_dist = ub - x

    lower_threshold = rtol * np.maximum(1, np.abs(lb))
    upper_threshold = rtol * np.maximum(1, np.abs(ub))

    lower_active = (np.isfinite(lb) &
                    (lower_dist <= np.minimum(upper_dist, lower_threshold)))
    active[lower_active] = -1

    upper_active = (np.isfinite(ub) &
                    (upper_dist <= np.minimum(lower_dist, upper_threshold)))
    active[upper_active] = 1

    return active


def make_strictly_feasible(x, lb, ub, rstep=1e-10):
    x_new = x.copy()

    active = find_active_constraints(x, lb, ub, rstep)
    lower_mask = np.equal(active, -1)
    upper_mask = np.equal(active, 1)

    if rstep == 0:
        x_new[lower_mask] = np.nextafter(lb[lower_mask], ub[lower_mask])
        x_new[upper_mask] = np.nextafter(ub[upper_mask], lb[upper_mask])
    else:
        x_new[lower_mask] = (lb[lower_mask] +
                             rstep * np.maximum(1, np.abs(lb[lower_mask])))
        x_new[upper_mask] = (ub[upper_mask] -
                             rstep * np.maximum(1, np.abs(ub[upper_mask])))

    tight_bounds = (x_new < lb) | (x_new > ub)
    x_new[tight_bounds] = 0.5 * (lb[tight_bounds] + ub[tight_bounds])

    return x_new


def CL_scaling_vector(x, g, lb, ub):
    v = np.ones_like(x)
    dv = np.zeros_like(x)

    mask = ((g < 0) & np.isfinite(ub))
    mask = np.squeeze(np.asarray(mask))
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
        return y, np.ones_like(y)

    lb_finite = np.isfinite(lb)
    ub_finite = np.isfinite(ub)

    x = y.copy()
    g_negative = np.zeros_like(y, dtype=bool)

    mask = lb_finite & ~ub_finite
    x[mask] = np.maximum(y[mask], 2 * lb[mask] - y[mask])
    g_negative[mask] = y[mask] < lb[mask]

    mask = ~lb_finite & ub_finite
    x[mask] = np.minimum(y[mask], 2 * ub[mask] - y[mask])
    g_negative[mask] = y[mask] > ub[mask]

    mask = lb_finite & ub_finite
    d = ub - lb
    t = np.remainder(y[mask] - lb[mask], 2 * d[mask])
    x[mask] = lb[mask] + np.minimum(t, 2 * d[mask] - t)
    g_negative[mask] = t > d[mask]

    g = np.ones_like(y)
    g[g_negative] = -1

    return x, g


# Simple helper functions.

def compute_grad(J, f):
    return J.T.dot(f)

