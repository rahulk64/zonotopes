import numpy as np

class LinearOperator(object):
    def __new__(cls, *args, **kwargs):
        if cls is LinearOperator:
            # Operate as _CustomLinearOperator factory.
            return super(LinearOperator, cls).__new__(_CustomLinearOperator)
        else:
            obj = super(LinearOperator, cls).__new__(cls)

            if (type(obj)._matvec == LinearOperator._matvec
                    and type(obj)._matmat == LinearOperator._matmat):
                warnings.warn("LinearOperator subclass should implement"
                              " at least one of _matvec and _matmat.",
                              category=RuntimeWarning, stacklevel=2)

            return obj

    def __init__(self, dtype, shape):
        if dtype is not None:
            dtype = np.dtype(dtype)

        shape = tuple(shape)
        if not isshape(shape):
            raise ValueError("invalid shape %r (must be 2-d)" % (shape,))

        self.dtype = dtype
        self.shape = shape

    def _init_dtype(self):
        if self.dtype is None:
            v = np.zeros(self.shape[-1])
            self.dtype = np.asarray(self.matvec(v)).dtype

    def _matmat(self, X):
        return np.hstack([self.matvec(col.reshape(-1,1)) for col in X.T])

    def _matvec(self, x):
        return self.matmat(x.reshape(-1, 1))

    def matvec(self, x):
        x = np.asanyarray(x)

        M,N = self.shape

        if x.shape != (N,) and x.shape != (N,1):
            raise ValueError('dimension mismatch')

        y = self._matvec(x)

        if isinstance(x, np.matrix):
            y = asmatrix(y)
        else:
            y = np.asarray(y)

        if x.ndim == 1:
            y = y.reshape(M)
        elif x.ndim == 2:
            y = y.reshape(M,1)
        else:
            raise ValueError('invalid shape returned by user-defined matvec()')

        return y

    def rmatvec(self, x):
        x = np.asanyarray(x)

        M,N = self.shape

        if x.shape != (M,) and x.shape != (M,1):
            raise ValueError('dimension mismatch')

        y = self._rmatvec(x)

        if isinstance(x, np.matrix):
            y = asmatrix(y)
        else:
            y = np.asarray(y)

        if x.ndim == 1:
            y = y.reshape(N)
        elif x.ndim == 2:
            y = y.reshape(N,1)
        else:
            raise ValueError('invalid shape returned by user-defined rmatvec()')

        return y

    def _rmatvec(self, x):
        if type(self)._adjoint == LinearOperator._adjoint:
            # _adjoint not overridden, prevent infinite recursion
            raise NotImplementedError
        else:
            return self.H.matvec(x)

    def matmat(self, X):
        X = np.asanyarray(X)

        if X.ndim != 2:
            raise ValueError('expected 2-d ndarray or matrix, not %d-d'
                             % X.ndim)

        if X.shape[0] != self.shape[1]:
            raise ValueError('dimension mismatch: %r, %r'
                             % (self.shape, X.shape))

        Y = self._matmat(X)

        if isinstance(Y, np.matrix):
            Y = asmatrix(Y)

        return Y

    def rmatmat(self, X):
        X = np.asanyarray(X)

        if X.ndim != 2:
            raise ValueError('expected 2-d ndarray or matrix, not %d-d'
                             % X.ndim)

        if X.shape[0] != self.shape[0]:
            raise ValueError('dimension mismatch: %r, %r'
                             % (self.shape, X.shape))

        Y = self._rmatmat(X)
        if isinstance(Y, np.matrix):
            Y = asmatrix(Y)
        return Y

    def _rmatmat(self, X):
        if type(self)._adjoint == LinearOperator._adjoint:
            return np.hstack([self.rmatvec(col.reshape(-1, 1)) for col in X.T])
        else:
            return self.H.matmat(X)

    def __call__(self, x):
        return self*x

    def __mul__(self, x):
        return self.dot(x)

    def dot(self, x):
        if isinstance(x, LinearOperator):
            return _ProductLinearOperator(self, x)
        elif np.isscalar(x):
            return _ScaledLinearOperator(self, x)
        else:
            x = np.asarray(x)

            if x.ndim == 1 or x.ndim == 2 and x.shape[1] == 1:
                return self.matvec(x)
            elif x.ndim == 2:
                return self.matmat(x)
            else:
                raise ValueError('expected 1-d or 2-d array or matrix, got %r'
                                 % x)

    def __matmul__(self, other):
        if np.isscalar(other):
            raise ValueError("Scalar operands are not allowed, "
                             "use '*' instead")
        return self.__mul__(other)

    def __rmatmul__(self, other):
        if np.isscalar(other):
            raise ValueError("Scalar operands are not allowed, "
                             "use '*' instead")
        return self.__rmul__(other)

    def __rmul__(self, x):
        if np.isscalar(x):
            return _ScaledLinearOperator(self, x)
        else:
            return NotImplemented

    def __pow__(self, p):
        if np.isscalar(p):
            return _PowerLinearOperator(self, p)
        else:
            return NotImplemented

    def __add__(self, x):
        if isinstance(x, LinearOperator):
            return _SumLinearOperator(self, x)
        else:
            return NotImplemented

    def __neg__(self):
        return _ScaledLinearOperator(self, -1)

    def __sub__(self, x):
        return self.__add__(-x)

    def __repr__(self):
        M,N = self.shape
        if self.dtype is None:
            dt = 'unspecified dtype'
        else:
            dt = 'dtype=' + str(self.dtype)

        return '<%dx%d %s with %s>' % (M, N, self.__class__.__name__, dt)

    def adjoint(self):
        return self._adjoint()

    H = property(adjoint)

    def transpose(self):
        return self._transpose()

    T = property(transpose)

    def _adjoint(self):
        return _AdjointLinearOperator(self)

    def _transpose(self):
        return _TransposedLinearOperator(self)


class _CustomLinearOperator(LinearOperator):
    def __init__(self, shape, matvec, rmatvec=None, matmat=None,
                 dtype=None, rmatmat=None):
        super(_CustomLinearOperator, self).__init__(dtype, shape)

        self.args = ()

        self.__matvec_impl = matvec
        self.__rmatvec_impl = rmatvec
        self.__rmatmat_impl = rmatmat
        self.__matmat_impl = matmat

        self._init_dtype()

    def _matmat(self, X):
        if self.__matmat_impl is not None:
            return self.__matmat_impl(X)
        else:
            return super(_CustomLinearOperator, self)._matmat(X)

    def _matvec(self, x):
        return self.__matvec_impl(x)

    def _rmatvec(self, x):
        func = self.__rmatvec_impl
        if func is None:
            raise NotImplementedError("rmatvec is not defined")
        return self.__rmatvec_impl(x)

    def _rmatmat(self, X):
        if self.__rmatmat_impl is not None:
            return self.__rmatmat_impl(X)
        else:
            return super(_CustomLinearOperator, self)._rmatmat(X)

    def _adjoint(self):
        return _CustomLinearOperator(shape=(self.shape[1], self.shape[0]),
                                     matvec=self.__rmatvec_impl,
                                     rmatvec=self.__matvec_impl,
                                     matmat=self.__rmatmat_impl,
                                     rmatmat=self.__matmat_impl,
                                     dtype=self.dtype)

class MatrixLinearOperator(LinearOperator):
    def __init__(self, A):
        super(MatrixLinearOperator, self).__init__(A.dtype, A.shape)
        self.A = A
        self.__adj = None
        self.args = (A,)

    def _matmat(self, X):
        return self.A.dot(X)

    def _adjoint(self):
        if self.__adj is None:
            self.__adj = _AdjointMatrixOperator(self)
        return self.__adj

class _AdjointMatrixOperator(MatrixLinearOperator):
    def __init__(self, adjoint):
        self.A = adjoint.A.T.conj()
        self.__adjoint = adjoint
        self.args = (adjoint,)
        self.shape = adjoint.shape[1], adjoint.shape[0]

    @property
    def dtype(self):
        return self.__adjoint.dtype

    def _adjoint(self):
        return self.__adjoint

def isshape(x, nonneg=False):
    try:
        # Assume it's a tuple of matrix dimensions (M, N)
        (M, N) = x
    except Exception:
        return False
    else:
        if np.ndim(M) == 0 and np.ndim(N) == 0:
            if not nonneg or (M >= 0 and N >= 0):
                return True
        return False

def aslinearoperator(A):
    if isinstance(A, LinearOperator):
        return A

    elif isinstance(A, np.ndarray) or isinstance(A, np.matrix):
        if A.ndim > 2:
            raise ValueError('array must have ndim <= 2')
        A = np.atleast_2d(np.asarray(A))
        return MatrixLinearOperator(A)

    else:
        if hasattr(A, 'shape') and hasattr(A, 'matvec'):
            rmatvec = None
            rmatmat = None
            dtype = None

            if hasattr(A, 'rmatvec'):
                rmatvec = A.rmatvec
            if hasattr(A, 'rmatmat'):
                rmatmat = A.rmatmat
            if hasattr(A, 'dtype'):
                dtype = A.dtype
            return LinearOperator(A.shape, A.matvec, rmatvec=rmatvec,
                                  rmatmat=rmatmat, dtype=dtype)

        else:
            raise TypeError('type not understood')
