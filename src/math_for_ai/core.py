"""Small educational numerical methods; real finite inputs only."""
import numpy as np


def array(x, ndim):
    if np.iscomplexobj(x):
        raise ValueError('Complex inputs are not supported')
    x = np.asarray(x, dtype=float)
    if x.ndim != ndim or not x.size or not np.isfinite(x).all():
        raise ValueError('Expected nonempty finite array of specified rank')
    return x


def lu(a):
    """Partial-pivoted LU: P @ A = L @ U; reject near-zero pivots."""
    u = array(a, 2).copy()
    n, m = u.shape
    if n != m:
        raise ValueError('Square matrix required')
    p, l = np.eye(n), np.eye(n)
    tol = np.finfo(float).eps * n * np.max(np.abs(u))
    for k in range(n):
        j = k + np.argmax(np.abs(u[k:, k]))
        if abs(u[j, k]) <= tol:
            raise ValueError('Numerically singular pivot')
        u[[k, j]] = u[[j, k]]
        p[[k, j]] = p[[j, k]]
        l[[k, j], :k] = l[[j, k], :k]
        for i in range(k + 1, n):
            l[i, k] = u[i, k] / u[k, k]
            u[i, k:] -= l[i, k] * u[k, k:]
            u[i, k] = 0
    return p, l, u


def qr(a):
    """Reduced modified Gram-Schmidt QR, full column rank only."""
    a = array(a, 2)
    m, n = a.shape
    if m < n:
        raise ValueError('Expected tall or square matrix')
    q, r = np.zeros((m, n)), np.zeros((n, n))
    tol = np.finfo(float).eps * max(m, n) * np.linalg.norm(a)
    for j in range(n):
        v = a[:, j].copy()
        for i in range(j):
            r[i, j] = q[:, i] @ v
            v -= r[i, j] * q[:, i]
        r[j, j] = np.linalg.norm(v)
        if r[j, j] <= tol:
            raise ValueError('Numerically rank deficient')
        q[:, j] = v / r[j, j]
    return q, r


def solve(a, b):
    p, l, u = lu(a)
    b = array(b, 1)
    n = len(l)
    if len(b) != n:
        raise ValueError('Incompatible target length')
    rhs = p @ b
    y, x = np.zeros(n), np.zeros(n)
    for i in range(n):
        y[i] = rhs[i] - l[i, :i] @ y[:i]
    for i in reversed(range(n)):
        x[i] = (y[i] - u[i, i+1:] @ x[i+1:]) / u[i, i]
    return x


def scalar_objective(f, x):
    value = np.asarray(f(x))
    if value.ndim != 0 or np.iscomplexobj(value):
        raise ValueError('Expected a real scalar objective')
    value = float(value)
    if not np.isfinite(value):
        raise FloatingPointError('Nonfinite objective')
    return value


def gradient(f, x, h=1e-5):
    """Central finite differences for scalar-valued functions."""
    x = array(x, 1)
    if not np.isfinite(h) or h <= 0:
        raise ValueError('Positive finite step required')
    out = np.zeros_like(x)
    for i in range(len(x)):
        delta = np.zeros_like(x)
        delta[i] = h
        plus, minus = x + delta, x - delta
        if not np.isfinite(plus).all() or not np.isfinite(minus).all():
            raise FloatingPointError('Nonfinite differentiation point')
        if plus[i] == x[i] or minus[i] == x[i]:
            raise ValueError('Step is too small to change the input')
        out[i] = (scalar_objective(f, plus) - scalar_objective(f, minus)) / (2*h)
    if not np.isfinite(out).all():
        raise FloatingPointError('Nonfinite derivative')
    return out


def optimize(f, grad, x, method='gd', rate=0.01, steps=200):
    """Fixed-step gradient descent, heavy-ball momentum, or Adam."""
    x = array(x, 1).copy()
    if method not in ('gd', 'momentum', 'adam') or not np.isfinite(rate) or rate <= 0 or type(steps) is not int or steps < 1:
        raise ValueError('Invalid optimizer configuration')
    m, v = np.zeros_like(x), np.zeros_like(x)
    history = [scalar_objective(f, x)]
    for t in range(1, steps+1):
        g = array(grad(x), 1)
        if g.shape != x.shape:
            raise ValueError('Gradient shape mismatch')
        if method == 'gd':
            update = g
        elif method == 'momentum':
            m = .9*m + g
            update = m
        else:
            with np.errstate(over='raise', invalid='raise'):
                m, v = .9*m + .1*g, .999*v + .001*g*g
            update = (m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8)
        with np.errstate(over='raise', invalid='raise'):
            x -= rate*update
        if not np.isfinite(x).all() or not np.isfinite(update).all():
            raise FloatingPointError('Nonfinite optimization state')
        history.append(scalar_objective(f, x))
    return x, np.asarray(history)


class Value:
    """Scalar reverse-mode autodiff. New graph for each forward evaluation."""
    def __init__(self, data, parents=()):
        self.data = float(data)
        if not np.isfinite(self.data):
            raise ValueError('Nonfinite value')
        self.parents = parents
        self.grad = 0.0

    @staticmethod
    def wrap(x):
        return x if isinstance(x, Value) else Value(x)

    def __add__(self, other):
        other = self.wrap(other)
        return Value(self.data + other.data, ((self, 1.), (other, 1.)))

    __radd__ = __add__

    def __mul__(self, other):
        other = self.wrap(other)
        return Value(self.data * other.data, ((self, other.data), (other, self.data)))

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -self.wrap(other)

    def __rsub__(self, other):
        return self.wrap(other) + -self

    def exp(self):
        val = np.exp(self.data)
        return Value(val, ((self, val),))

    def tanh(self):
        val = np.tanh(self.data)
        return Value(val, ((self, 1-val*val),))

    def backward(self):
        order, visited = [], set()
        def visit(node):
            if node in visited:
                return
            visited.add(node)
            for parent, _ in node.parents:
                visit(parent)
            order.append(node)
        visit(self)
        for node in order:
            node.grad = 0.
        self.grad = 1.
        for node in reversed(order):
            for parent, weight in node.parents:
                parent.grad += weight*node.grad
