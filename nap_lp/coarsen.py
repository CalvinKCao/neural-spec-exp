"""Coarsen (Algorithm 1) and release orders."""
import time
import numpy as np

from .net import STAR
from .relax import VERIFIED, NOT_VERIFIED, TIMEOUT


def forward_order(net, P):
    return [int(g) for g in range(net.N) if P[g] != STAR]


def backward_order(net, P):
    """Layers last to first, neuron index ascending within a layer."""
    out = []
    for l in reversed(range(net.L)):
        out += [int(g) for g in range(net.off[l], net.off[l + 1]) if P[g] != STAR]
    return out


def random_order(net, P, rng):
    o = forward_order(net, P)
    rng.shuffle(o)
    return o


def dual_order(net, P, relax):
    """Release pins with the smallest |dual| first (ties: forward order). Returns (order, duals or None)."""
    out = relax.solve(P)
    fo = forward_order(net, P)
    if out.duals is None:  # LP infeasible/empty -> no duals; caller should note it
        return fo, None
    d = out.duals
    return sorted(fo, key=lambda g: (d[g], g)), d


def coarsen(verifier, mined, order, diag=True):
    """Returns dict(pattern=None if mined does not verify, steps=[...], n_calls, timeouts, time).
    Calls verifier.verify; a TIMEOUT counts as NOT_VERIFIED (pin stays). Every step is logged."""
    t0 = time.perf_counter()
    P = mined.copy()
    n_log0 = len(verifier.log)
    v0 = verifier.verify(P)
    res = dict(mined_verdict=v0, steps=[], pattern=None)
    if v0 != VERIFIED:
        res.update(n_calls=1, timeouts=int(v0 == TIMEOUT), time=time.perf_counter() - t0,
                   call_log=verifier.log[n_log0:])
        return res
    for g in order:
        old = P[g]
        assert old != STAR
        P[g] = STAR
        v = verifier.verify(P)
        released = v == VERIFIED
        step = dict(tried=P.copy(),neuron=g, state=int(old), verdict=v, released=released, time=verifier.log[-1]["time"],
                    stage=verifier.log[-1].get("stage"))
        if released and diag:
            b = verifier.relax.neuron_bounds(P, g)
            if b is None:
                step["implied"] = True  # region already empty
            else:
                lo, hi = b
                step["implied"] = bool(lo >= -1e-9) if old == 1 else bool(hi <= 1e-9)
        if not released:
            P[g] = old
        res["steps"].append(step)
    res["pattern"] = P
    log = verifier.log[n_log0:]
    res.update(n_calls=len(log), timeouts=sum(1 for e in log if e["verdict"] == TIMEOUT),
               time=time.perf_counter() - t0, call_log=log)
    return res
