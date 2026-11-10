"""Plain float64 ReLU network, patterns, mining, coverage.

A pattern is a flat int8 array over all hidden neurons: 1 (post-act > 0),
0 (post-act == 0), -1 (* / unconstrained).
"""
import numpy as np

STAR = -1


class Net:
    """Feed-forward ReLU net. Ws/bs: list of arrays, last pair is the output layer."""

    def __init__(self, Ws, bs):
        self.W = [np.asarray(w, dtype=np.float64) for w in Ws]
        self.b = [np.asarray(b, dtype=np.float64) for b in bs]
        self.L = len(Ws) - 1  # hidden layers
        self.n_in = self.W[0].shape[1]
        self.widths = [w.shape[0] for w in self.W[:-1]]
        self.off = np.concatenate([[0], np.cumsum(self.widths)]).astype(int)
        self.N = int(self.off[-1])
        assert self.W[-1].shape[0] == 2, "binary margin only"

    def margin_vec(self, c):
        """Margin = wm . a_L + bm  (logit_c - logit_other)."""
        o = 1 - c
        return self.W[-1][c] - self.W[-1][o], float(self.b[-1][c] - self.b[-1][o])

    def forward(self, X):
        X = np.atleast_2d(X)
        zs, a = [], X
        for l in range(self.L):
            z = a @ self.W[l].T + self.b[l]
            zs.append(z)
            a = np.maximum(z, 0.0)
        logits = a @ self.W[-1].T + self.b[-1]
        return zs, logits

    def margin(self, X, c):
        _, lg = self.forward(X)
        return lg[:, c] - lg[:, 1 - c]

    def preacts_flat(self, X):
        zs, _ = self.forward(X)
        return np.concatenate(zs, axis=1)  # (n, N)

    def predict(self, X):
        return self.forward(X)[1].argmax(1)

    def layer_of(self, g):
        l = int(np.searchsorted(self.off, g, side="right") - 1)
        return l, int(g - self.off[l])


def full_pattern(net, x):
    """Fully pinned pattern of a single input."""
    z = net.preacts_flat(x)[0]
    return (z > 0).astype(np.int8)


def mine(net, X, y, c, delta=0.95, correct_only=False):
    """Class-c NAP by vote. X,y: training data. 1 if >= delta of inputs have z>0, 0 if >= delta have z<=0."""
    mask = y == c
    if correct_only:
        mask &= net.predict(X) == c
    Z = net.preacts_flat(X[mask])
    n = Z.shape[0]
    need = int(np.ceil(delta * n - 1e-9))
    pos = (Z > 0).sum(0)
    neg = n - pos
    P = np.full(net.N, STAR, dtype=np.int8)
    P[pos >= need] = 1
    P[neg >= need] = 0
    return P


def covered(net, X, P):
    """Boolean mask of inputs whose actual pattern matches every pinned neuron of P."""
    Z = net.preacts_flat(X)
    ok = np.ones(Z.shape[0], dtype=bool)
    p1 = P == 1
    p0 = P == 0
    ok &= (Z[:, p1] > 0).all(1)
    ok &= (Z[:, p0] <= 0).all(1)
    return ok


def per_layer_counts(net, P):
    return [int((P[net.off[l]:net.off[l + 1]] != STAR).sum()) for l in range(net.L)]
