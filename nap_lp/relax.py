"""LP triangle relaxation with LP-derived bounds, plus the shared bound engine used by the MILP verifier.

Variable layout: x (n_in), then for hidden layer l: z_l (w_l) followed by a_l (w_l).
Bounds for layer l are LP min/max of z_l over the constraints of layers 0..l-1 only
(their pins and relaxations). Pins of layer l or later are not used (unless retighten > 0).
"""
import time
from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from .net import STAR

VERIFIED, NOT_VERIFIED, TIMEOUT = "VERIFIED", "NOT_VERIFIED", "TIMEOUT"
THR = 1e-6   # property: margin > THR
EPS = 1e-8   # outward widening of computed bounds (numerical soundness)

INACTIVE, ACTIVE, UNSTABLE, PINNED = 1, 2, 3, 0


@dataclass
class Bounds:
    empty: bool
    lo: np.ndarray = None    # raw bounds, nan for pinned neurons
    hi: np.ndarray = None
    kind: np.ndarray = None  # PINNED / INACTIVE / ACTIVE / UNSTABLE


@dataclass
class LPOut:
    verdict: str
    opt: float = np.nan        # min margin (nan if empty)
    x: np.ndarray = None       # minimizer
    empty: bool = False
    duals: np.ndarray = None   # |dual| of the pin sign constraint per hidden neuron (0 for unpinned)
    raw_duals: np.ndarray = None


class Relaxation:
    def __init__(self, net, c, cache=True, prefilter=True, retighten=0):
        self.net, self.c = net, c
        self.use_cache, self.prefilter, self.retighten = cache, prefilter, retighten
        if retighten:
            self.use_cache = False  # key would have to be the whole pattern; spec: cache off
        self.wm, self.bm = net.margin_vec(c)
        self.cache = {}
        self.reset_stats()
        self.Wp = [np.maximum(w, 0) for w in net.W]
        self.Wn = [np.minimum(w, 0) for w in net.W]

    def reset_stats(self):
        self.stats = dict(bound_lp=0, bound_lp_skipped=0, cache_lookups=0, cache_hits=0,
                          final_lp=0, t_bounds=0.0, t_final=0.0, empty=0)

    def clear_cache(self):
        self.cache = {}

    # ---------- constraint system ----------
    def _z(self, l):
        return self.net.n_in + 2 * int(self.net.off[l])

    def _a(self, l):
        return self._z(l) + self.net.widths[l]

    def nvar(self, upto):
        return self.net.n_in + 2 * int(self.net.off[upto])

    def _system(self, pattern, lo, hi, kind, upto, milp=False, margin_le=None):
        """Constraints of layers 0..upto-1. Returns dict with sparse matrices etc.
        milp=True: unstable neurons get big-M binaries instead of the triangle.
        margin_le: if given (needs upto == L) add row  margin <= margin_le."""
        net, off = self.net, self.net.off
        nv0 = self.nvar(upto)
        lb = np.full(nv0, -np.inf)
        ub = np.full(nv0, np.inf)
        lb[:net.n_in], ub[:net.n_in] = 0.0, 1.0
        eq = [[], [], [], []]  # rows, cols, vals, rhs (rhs per row)
        le = [[], [], [], []]
        neq = [0]
        nle = [0]
        nbin = [0]
        bin_cols = []
        pin_rows = {}

        def add(store, cnt, cols, vals, rhs):
            r = cnt[0]
            store[0].extend([r] * len(cols))
            store[1].extend(cols)
            store[2].extend(vals)
            store[3].append(rhs)
            cnt[0] += 1
            return r

        for l in range(upto):
            w = net.widths[l]
            zo, ao = self._z(l), self._a(l)
            prev = np.arange(net.n_in) if l == 0 else np.arange(self._a(l - 1), self._a(l - 1) + net.widths[l - 1])
            W, b = net.W[l], net.b[l]
            lb[ao:ao + w] = 0.0  # a = ReLU(z) >= 0 always
            for j in range(w):
                g = off[l] + j
                zj, aj = zo + j, ao + j
                add(eq, neq, [zj] + prev.tolist(), [1.0] + (-W[j]).tolist(), float(b[j]))
                p = pattern[g]
                if p == 1:
                    pin_rows[g] = add(le, nle, [zj], [-1.0], 0.0)
                    add(eq, neq, [aj, zj], [1.0, -1.0], 0.0)
                elif p == 0:
                    pin_rows[g] = add(le, nle, [zj], [1.0], 0.0)
                    ub[aj] = 0.0
                else:
                    k = kind[g]
                    if k == INACTIVE:
                        ub[aj] = 0.0
                    elif k == ACTIVE:
                        add(eq, neq, [aj, zj], [1.0, -1.0], 0.0)
                    else:
                        lw, hw = lo[g] - EPS, hi[g] + EPS
                        add(le, nle, [zj, aj], [1.0, -1.0], 0.0)  # a >= z
                        if milp:
                            d = nv0 + nbin[0]
                            nbin[0] += 1
                            bin_cols.append(d)
                            add(le, nle, [aj, zj, d], [1.0, -1.0, -lw], -lw)  # a <= z - lw(1-d)
                            add(le, nle, [aj, d], [1.0, -hw], 0.0)          # a <= hw d
                        else:
                            s = hw / (hw - lw)
                            add(le, nle, [aj, zj], [1.0, -s], -s * lw)       # a <= s (z - lw)
        if margin_le is not None:
            ao = self._a(net.L - 1)
            add(le, nle, list(range(ao, ao + net.widths[-1])), self.wm.tolist(), margin_le - self.bm)
        nv = nv0 + nbin[0]
        if nbin[0]:
            lb = np.concatenate([lb, np.zeros(nbin[0])])
            ub = np.concatenate([ub, np.ones(nbin[0])])
        mk = lambda st, n: (sp.csr_matrix((st[2], (st[0], st[1])), shape=(n, nv)), np.array(st[3], dtype=float)) if n else (None, None)
        Aeq, beq = mk(eq, neq[0])
        Ale, ble = mk(le, nle[0])
        integ = np.zeros(nv)
        integ[bin_cols] = 1
        return dict(Aeq=Aeq, beq=beq, Ale=Ale, ble=ble, lb=lb, ub=ub, nv=nv, pin_rows=pin_rows, integ=integ)

    @property
    def wm_(self):
        return self.wm

    def _lp(self, S, cvec, want_duals=False):
        res = linprog(cvec, A_ub=S["Ale"], b_ub=S["ble"], A_eq=S["Aeq"], b_eq=S["beq"],
                      bounds=np.column_stack([S["lb"], S["ub"]]), method="highs")
        if res.status not in (0, 2):
            raise RuntimeError(f"LP status {res.status}: {res.message}")
        return res

    # ---------- bounds ----------
    def _lp_layer_bounds(self, l, pattern, lo, hi, kind, need, ent):
        """LP min/max of z_l[j] over layers < l for neurons in `need`. Returns False if infeasible."""
        net = self.net
        S = self._system(pattern, lo, hi, kind, l)
        prev = np.arange(net.n_in) if l == 0 else np.arange(self._a(l - 1), self._a(l - 1) + net.widths[l - 1])
        for j in np.flatnonzero(need):
            cv = np.zeros(S["nv"])
            cv[prev] = net.W[l][j]
            r1 = self._lp(S, cv)
            if r1.status == 2:
                return False
            r2 = self._lp(S, -cv)
            self.stats["bound_lp"] += 2
            ent["lo"][j] = r1.fun + net.b[l][j]
            ent["hi"][j] = -r2.fun + net.b[l][j]
        return True

    def bounds(self, pattern):
        t0 = time.perf_counter()
        net, off, L = self.net, self.net.off, self.net.L
        lo = np.full(net.N, np.nan)
        hi = np.full(net.N, np.nan)
        kind = np.zeros(net.N, dtype=np.int8)
        a_lo, a_hi = np.zeros(net.n_in), np.ones(net.n_in)
        empty = Bounds(True)
        for l in range(L):
            sl = slice(off[l], off[l + 1])
            pat = pattern[sl]
            key = (l, pattern[:off[l]].tobytes())
            ent = self.cache.get(key) if self.use_cache else None
            self.stats["cache_lookups"] += 1
            if ent is None:
                ent = dict(lo=np.full(net.widths[l], np.nan), hi=np.full(net.widths[l], np.nan), empty=False)
                if self.use_cache:
                    self.cache[key] = ent
            else:
                self.stats["cache_hits"] += 1
            if ent["empty"]:
                self.stats["empty"] += 1
                self.stats["t_bounds"] += time.perf_counter() - t0
                return empty
            W, b = net.W[l], net.b[l]
            zlo = self.Wp[l] @ a_lo + self.Wn[l] @ a_hi + b
            zhi = self.Wp[l] @ a_hi + self.Wn[l] @ a_lo + b
            if self.prefilter:
                # pinned neuron whose interval contradicts its pin: region is empty
                if ((pat == 1) & (zhi < -1e-6)).any() or ((pat == 0) & (zlo > 1e-6)).any():
                    self.stats["empty"] += 1
                    self.stats["t_bounds"] += time.perf_counter() - t0
                    return empty
            need = (pat == STAR) & np.isnan(ent["lo"])
            if need.any() and self.prefilter:
                stab = need & ((zhi + EPS <= 0) | (zlo - EPS >= 0))
                ent["lo"][stab], ent["hi"][stab] = zlo[stab], zhi[stab]
                self.stats["bound_lp_skipped"] += 2 * int(stab.sum())
                need = need & ~stab
            if need.any():
                if not self._lp_layer_bounds(l, pattern, lo, hi, kind, need, ent):
                    ent["empty"] = True
                    self.stats["empty"] += 1
                    self.stats["t_bounds"] += time.perf_counter() - t0
                    return empty
            un = pat == STAR
            lo[sl][un] = ent["lo"][un]
            hi[sl][un] = ent["hi"][un]
            self._classify(lo, hi, kind, pattern, l)
            a_lo, a_hi = self._next_interval(pattern, lo, hi, kind, l, zlo, zhi)
        res = Bounds(False, lo, hi, kind)
        if self.retighten:
            res = self._retighten(pattern, res)
        self.stats["t_bounds"] += time.perf_counter() - t0
        return res

    def _classify(self, lo, hi, kind, pattern, l):
        off = self.net.off
        for j in range(self.net.widths[l]):
            g = off[l] + j
            if pattern[g] != STAR:
                kind[g] = PINNED
            elif hi[g] + EPS <= 0:
                kind[g] = INACTIVE
            elif lo[g] - EPS >= 0:
                kind[g] = ACTIVE
            else:
                kind[g] = UNSTABLE

    def _next_interval(self, pattern, lo, hi, kind, l, zlo, zhi):
        off = self.net.off
        sl = slice(off[l], off[l + 1])
        pat = pattern[sl]
        al = np.zeros(self.net.widths[l])
        ah = np.zeros(self.net.widths[l])
        un = pat == STAR
        al[un] = np.maximum(lo[sl][un] - EPS, 0)
        ah[un] = np.maximum(hi[sl][un] + EPS, 0)
        p1 = pat == 1
        al[p1] = np.maximum(zlo[p1], 0)
        ah[p1] = np.maximum(zhi[p1], 0)
        return al, ah  # pinned 0 stays [0,0]

    def _retighten(self, pattern, bd):
        """Ablation: re-tighten unstable neurons' bounds with LPs over the whole model incl. later pins."""
        net, off, L = self.net, self.net.off, self.net.L
        lo, hi, kind = bd.lo.copy(), bd.hi.copy(), bd.kind.copy()
        for _ in range(self.retighten):
            for l in range(L):
                S = self._system(pattern, lo, hi, kind, L)
                zo = self._z(l)
                for j in range(net.widths[l]):
                    g = off[l] + j
                    if kind[g] != UNSTABLE:
                        continue
                    cv = np.zeros(S["nv"])
                    cv[zo + j] = 1.0
                    r1 = self._lp(S, cv)
                    if r1.status == 2:
                        return Bounds(True)
                    r2 = self._lp(S, -cv)
                    self.stats["bound_lp"] += 2
                    lo[g] = max(lo[g], r1.fun)
                    hi[g] = min(hi[g], -r2.fun)
                self._classify(lo, hi, kind, pattern, l)
        return Bounds(False, lo, hi, kind)

    # ---------- final margin LP ----------
    def solve(self, pattern, bd=None):
        pattern = np.asarray(pattern, dtype=np.int8)
        if bd is None:
            bd = self.bounds(pattern)
        if bd.empty:
            return LPOut(VERIFIED, empty=True)
        t0 = time.perf_counter()
        net = self.net
        S = self._system(pattern, bd.lo, bd.hi, bd.kind, net.L)
        cv = np.zeros(S["nv"])
        ao = self._a(net.L - 1)
        cv[ao:ao + net.widths[-1]] = self.wm
        res = self._lp(S, cv)
        self.stats["final_lp"] += 1
        self.stats["t_final"] += time.perf_counter() - t0
        if res.status == 2:
            self.stats["empty"] += 1
            return LPOut(VERIFIED, empty=True)
        opt = res.fun + self.bm
        raw = np.zeros(net.N)
        mg = res.ineqlin.marginals
        for g, r in S["pin_rows"].items():
            raw[g] = mg[r]
        return LPOut(VERIFIED if opt > THR else NOT_VERIFIED, opt=opt, x=res.x[:net.n_in].copy(),
                     duals=np.abs(raw), raw_duals=raw)

    def verify(self, pattern):
        return self.solve(pattern).verdict

    def neuron_bounds(self, pattern, g):
        """Raw LP bounds (layers < l) of neuron g under `pattern` (g should be unpinned there)."""
        bd = self.bounds(np.asarray(pattern, dtype=np.int8))
        if bd.empty:
            return None
        return float(bd.lo[g]), float(bd.hi[g])
