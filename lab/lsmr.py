import numpy as np
import sys
import tensorflow as tf

# NOTE AND REMINDER TO INCLUDE COPYRIGHT NOTICES

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

from numpy import zeros, infty, atleast_1d, result_type
from numpy.linalg import norm
from math import sqrt

def _sym_ortho(a, b):
    """
    Stable implementation of Givens rotation.
    Notes
    -----
    The routine 'SymOrtho' was added for numerical stability. This is
    recommended by S.-C. Choi in [1]_.  It removes the unpleasant potential of
    ``1/eps`` in some important places (see, for example text following
    "Compute the next plane rotation Qk" in minres.py).
    References
    ----------
    .. [1] S.-C. Choi, "Iterative Methods for Singular Linear Equations
           and Least-Squares Problems", Dissertation,
           http://www.stanford.edu/group/SOL/dissertations/sou-cheng-choi-thesis.pdf
    """
    if b == 0:
        return np.sign(a), 0, abs(a)
    elif a == 0:
        return 0, np.sign(b), abs(b)
    elif abs(b) > abs(a):
        tau = a / b
        s = np.sign(b) / sqrt(1 + tau * tau)
        c = s * tau
        r = b / s
    else:
        tau = b / a
        c = np.sign(a) / sqrt(1+tau*tau)
        s = c * tau
        r = a / c
    return c, s, r

def matmat(A, X):
    #X = np.asanyarray(X)
    if tf.rank(X) != 2:
        print("X shape", X.shape)
        tf.print("x:", X, output_stream=sys.stdout)
        tf.print("x rank:", tf.rank(X), output_stream=sys.stdout)
        #raise ValueError('expected 2-d ndarray or matrix, not', tf.rank(X))

    if X.shape[0] != A.shape[1]:
        raise ValueError('dimension mismatch: %r, %r'
                      % (A.shape, X.shape))

    #Y = self._matmat(X)
    #Y = A.dot(X)
    Y = tf.tensordot(A, X, axes=1)

    #if isinstance(Y, np.matrix):
    #    Y = asmatrix(Y)

    return Y

def matvec(A, x):
    M,N = A.shape

    #x = np.asanyarray(x)

    if x.shape != (N,) and x.shape != (N,1):
        raise ValueError('dimension mismatch: %r, %r'
                      % (A.shape, x.shape))

    #y = self._matvec(x)
    #y = A.matmat(x.reshape(-1, 1))
    #y = matmat(A, x.reshape(-1, 1))
    y = matmat(A, tf.reshape(x, (-1, 1)))

    #if isinstance(x, np.matrix):
    #    y = asmatrix(y)
    #else:
    #    y = np.asarray(y)

    if tf.rank(x) == 1:
        #y = y.reshape(M)
        y = tf.reshape(y, (M,))
    elif tf.rank(x) == 2:
        #y = y.reshape(M,1)
        y = tf.reshape(y, (M,1))
    else:
        #raise ValueError('invalid shape returned by user-defined matvec()')
        print('invalid shape returned by user-defined matvec()')

    return y

def rmatvec(A, x):
    m, n = A.shape
    x1 = x[:m]
    x2 = x[m:]

    #x = np.asanyarray(x)

    M,N = A.shape

    if x.shape != (M,) and x.shape != (M,1):
        raise ValueError('dimension mismatch: %r, %r'
                      % (A.shape, x.shape))

    #y = self._rmatvec(x)
    #y = matvec(A.H, x)
    y = matvec(tf.linalg.adjoint(A), x)

    #if isinstance(x, np.matrix):
    #    y = asmatrix(y)
    #else:
    #    y = np.asarray(y)

    if tf.rank(x)  == 1:
        #y = y.reshape(N)
        y = tf.reshape(y, (N,))
    elif tf.rank(x)  == 2:
        #y = y.reshape(N,1)
        y = tf.reshape(y, (N,1))
    else:
        #raise ValueError('invalid shape returned by user-defined rmatvec()')
        print('invalid shape returned by user-defined rmatvec()')

    return y

def lsmr(A, b, dis=None, diag=None, damp=0.0, atol=1e-6, btol=1e-6, conlim=1e8,
         maxiter=None, show=False, x0=None):

    #A = aslinearoperator(A)
    #b = atleast_1d(b)
    if tf.rank(b) > 1:
        b = tf.squeeze(b) 

    m, n = A.shape

    # stores the num of singular values
    if diag is not None:
        minDim = min([m+n, n])
    else:
        minDim = min([m, n])

    if maxiter is None:
        maxiter = minDim

    #dtype = result_type(A, b, float)
    dtype = tf.float64

    u = b
    #normb = norm(b)
    normb = tf.norm(b, ord='euclidean')
    if x0 is None:
        x = tf.zeros(n, dtype=tf.float64)
        #beta = normb.copy()
        beta = tf.identity(normb)
    else:
        print("this is not technically supported but it might work")
        x = atleast_1d(x0)
        #u = u - A.matvec(x)
        y = matvec(A, x)

        if diag is not None:
            rmo = matvec(A, np.ravel(x) * dis)
            y = np.hstack((rmo, diag * x))
            u = u - y
        else:
            u = u - y
        beta = norm(u)

    if beta > 0:
        u = (1 / beta) * u
        #v = A.rmatvec(u)

        if diag is not None:
            x1 = u[:m]
            x2 = u[m:]
            rmo = dis * rmatvec(A, x1)
            v = rmo + diag * x2
        else:
            v = rmatvec(A, u)
        alpha = tf.norm(v, ord='euclidean')
    else:
        v = zeros(n, dtype)
        alpha = 0

    if alpha > 0:
        v = (1 / alpha) * v

    # Initialize variables for 1st iteration.

    itn = 0
    zetabar = alpha * beta
    alphabar = alpha
    rho = 1
    rhobar = 1
    cbar = 1
    sbar = 0

    h = v.copy()
    hbar = zeros(n, dtype)

    # Initialize variables for estimation of ||r||.

    betadd = beta
    betad = 0
    rhodold = 1
    tautildeold = 0
    thetatilde = 0
    zeta = 0
    d = 0

    # Initialize variables for estimation of ||A|| and cond(A)

    normA2 = alpha * alpha
    maxrbar = 0
    minrbar = 1e+100
    normA = sqrt(normA2)
    condA = 1
    normx = 0

    # Items for use in stopping rules, normb set earlier
    istop = 0
    ctol = 0
    if conlim > 0:
        ctol = 1 / conlim
    normr = beta

    # Reverse the order here from the original matlab code because
    # there was an error on return when arnorm==0
    normar = alpha * beta
    if normar == 0:
        if show:
            print(msg[0])
        return x, istop, itn, normr, normar, normA, condA, normx

    # Main iteration loop.
    while itn < maxiter:
        itn = itn + 1

        # Perform the next step of the bidiagonalization to obtain the
        # next  beta, u, alpha, v.  These satisfy the relations
        #         beta*u  =  a*v   -  alpha*u,
        #        alpha*v  =  A'*u  -  beta*v.

        u *= -alpha
        #u += A.matvec(v)
        myvar = matvec(A, v)
        myvar = np.squeeze(np.asarray(myvar))
        if diag is not None:
            rmo = matvec(A, np.ravel(v) * dis)
            y = np.hstack((rmo, diag * v))
            u += y
        else:
            u += myvar
        beta = tf.norm(u, ord='euclidean')

        if beta > 0:
            u *= (1 / beta)
            v *= -beta
            #v += A.rmatvec(u)
            if diag is not None:
                x1 = u[:m]
                x2 = u[m:]
                rmo = dis * rmatvec(A, x1)
                y = rmo + diag * x2
                v += y
            else:
                v += rmatvec(A, u)

            alpha = tf.norm(v, ord='euclidean')
            if alpha > 0:
                v *= (1 / alpha)

        # At this point, beta = beta_{k+1}, alpha = alpha_{k+1}.

        # Construct rotation Qhat_{k,2k+1}.

        chat, shat, alphahat = _sym_ortho(alphabar, damp)

        # Use a plane rotation (Q_i) to turn B_i to R_i

        rhoold = rho
        c, s, rho = _sym_ortho(alphahat, beta)
        thetanew = s*alpha
        alphabar = c*alpha

        # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar

        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _sym_ortho(cbar * rho, thetanew)
        zeta = cbar * zetabar
        zetabar = - sbar * zetabar

        # Update h, h_hat, x.

        hbar *= - (thetabar * rho / (rhoold * rhobarold))
        hbar += h
        x += (zeta / (rho * rhobar)) * hbar
        h *= - (thetanew / rho)
        h += v

        # Estimate of ||r||.

        # Apply rotation Qhat_{k,2k+1}.
        betaacute = chat * betadd
        betacheck = -shat * betadd

        # Apply rotation Q_{k,k+1}.
        betahat = c * betaacute
        betadd = -s * betaacute

        # Apply rotation Qtilde_{k-1}.
        # betad = betad_{k-1} here.

        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = _sym_ortho(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = - stildeold * betad + ctildeold * betahat

        # betad   = betad_k here.
        # rhodold = rhod_k  here.

        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d = d + betacheck * betacheck
        normr = sqrt(d + (betad - taud)**2 + betadd * betadd)

        # Estimate ||A||.
        normA2 = normA2 + beta * beta
        normA = sqrt(normA2)
        normA2 = normA2 + alpha * alpha

        # Estimate cond(A).
        maxrbar = max(maxrbar, rhobarold)
        if itn > 1:
            minrbar = min(minrbar, rhobarold)
        condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)

        # Test for convergence.

        # Compute norms for convergence testing.
        normar = abs(zetabar)
        normx = norm(x)

        # Now use these norms to estimate certain other quantities,
        # some of which will be small near a solution.

        test1 = normr / normb
        if (normA * normr) != 0:
            test2 = normar / (normA * normr)
        else:
            test2 = infty
        test3 = 1 / condA
        t1 = test1 / (1 + normA * normx / normb)
        rtol = btol + atol * normA * normx / normb

        # The following tests guard against extremely small values of
        # atol, btol or ctol.  (The user may have set any or all of
        # the parameters atol, btol, conlim  to 0.)
        # The effect is equivalent to the normAl tests using
        # atol = eps,  btol = eps,  conlim = 1/eps.

        if itn >= maxiter:
            istop = 7
        if 1 + test3 <= 1:
            istop = 6
        if 1 + test2 <= 1:
            istop = 5
        if 1 + t1 <= 1:
            istop = 4

        # Allow for tolerances set by the user.

        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1

        if istop > 0:
            break

    return x, istop, itn, normr, normar, normA, condA, normx

