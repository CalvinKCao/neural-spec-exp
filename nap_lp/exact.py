"""Exact verifiers: reference MILP (HiGHS big-M, scipy.optimize.milp), cascade variant, shared interface.

All verifiers: verify(pattern) -> VERIFIED / NOT_VERIFIED / TIMEOUT, and keep .log (one dict per call).
"""
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds as SpBounds
import scipy.sparse as sp

from .relax import Relaxation, VERIFIED, NOT_VERIFIED, TIMEOUT, THR
from .net import STAR


class LPVerifier:
    mode = "LP"

    def __init__(self, net, c, cache=True, prefilter=True, retighten=0):
        self.net, self.c = net, c
        self.relax = Relaxation(net, c, cache=cache, prefilter=prefilter, retighten=retighten)
        self.log = []
        self.mode = "LP" if (cache or prefilter) and not retighten else ("LP-plain" if not retighten else f"LP-retighten{retighten}")

    def verify(self, pattern):
        t0 = time.perf_counter()
        out = self.relax.solve(pattern)
        self.log.append(dict(verdict=out.verdict, time=time.perf_counter() - t0, stage="lp", empty=out.empty, opt=out.opt))
        return out.verdict


class MILPVerifier:
    """mode_ = 'feasibility' (zero objective + margin <= THR; reference exact verifier used in Coarsen)
    or 'optimize' (minimise margin)."""

    def __init__(self, net, c, time_limit=300.0, mode="feasibility", cache=True, prefilter=True):
        self.net, self.c, self.time_limit, self.mode_ = net, c, time_limit, mode
        self.relax = Relaxation(net, c, cache=cache, prefilter=prefilter)
        self.log = []
        self.mode = f"MILP-{mode}"

    def solve(self, pattern, mode=None):
        """Returns (verdict, optimum_or_nan, x_or_None, empty)."""
        mode = mode or self.mode_
        pattern = np.asarray(pattern, dtype=np.int8)
        net, R = self.net, self.relax
        bd = R.bounds(pattern)
        if bd.empty:
            return VERIFIED, np.nan, None, True
        feas = mode == "feasibility"
        S = R._system(pattern, bd.lo, bd.hi, bd.kind, net.L, milp=True, margin_le=(THR if feas else None))
        cv = np.zeros(S["nv"])
        if not feas:
            ao = R._a(net.L - 1)
            cv[ao:ao + net.widths[-1]] = R.wm
        cons = []
        if S["Ale"] is not None:
            cons.append(LinearConstraint(S["Ale"], -np.inf, S["ble"]))
        if S["Aeq"] is not None:
            cons.append(LinearConstraint(S["Aeq"], S["beq"], S["beq"]))
        t0 = time.perf_counter()
        res = milp(cv, constraints=cons, integrality=S["integ"], bounds=SpBounds(S["lb"], S["ub"]),
                   options=dict(time_limit=self.time_limit, disp=False))
        R.stats["final_lp"] += 1
        R.stats["t_final"] += time.perf_counter() - t0
        if res.status == 2:
            return VERIFIED, np.nan, None, True
        if res.status == 0:
            x = res.x[:net.n_in].copy()
            if feas:
                return NOT_VERIFIED, np.nan, x, False
            opt = res.fun + R.bm
            return (VERIFIED if opt > THR else NOT_VERIFIED), opt, x, False
        if res.status == 1:
            if feas and res.x is not None:   # time limit but a feasible point (counterexample) exists
                return NOT_VERIFIED, np.nan, res.x[:net.n_in].copy(), False
            return TIMEOUT, np.nan, None, False
        raise RuntimeError(f"MILP status {res.status}: {res.message}")

    def verify(self, pattern):
        t0 = time.perf_counter()
        v, opt, x, empty = self.solve(pattern)
        self.log.append(dict(verdict=v, time=time.perf_counter() - t0, stage="milp", empty=empty, opt=opt,
                             timeout=(v == TIMEOUT)))
        return v


class CascadeVerifier:
    """VARIANT (not the reference): LP -> genuine-counterexample check on the LP minimiser -> reference MILP."""
    mode = "cascade"

    def __init__(self, net, c, time_limit=300.0):
        self.net, self.c = net, c
        self.milp = MILPVerifier(net, c, time_limit=time_limit, mode="feasibility")
        self.relax = self.milp.relax  # shared bounds engine (and cache)
        self.log = []

    def verify(self, pattern):
        t0 = time.perf_counter()
        pattern = np.asarray(pattern, dtype=np.int8)
        net = self.net
        out = self.relax.solve(pattern)
        if out.verdict == VERIFIED:
            v, stage = VERIFIED, "lp"
        else:
            x = np.clip(out.x, 0.0, 1.0)
            zs, lg = net.forward(x)
            z = np.concatenate([zz[0] for zz in zs])
            ok = bool((z[pattern == 1] >= 0).all() and (z[pattern == 0] <= 0).all())
            if ok and (lg[0, self.c] - lg[0, 1 - self.c]) <= THR:
                v, stage = NOT_VERIFIED, "lp_cex"
            else:
                v, _, _, _ = self.milp.solve(pattern)
                stage = "milp"
        self.log.append(dict(verdict=v, time=time.perf_counter() - t0, stage=stage, timeout=(v == TIMEOUT)))
        return v
