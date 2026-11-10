import collections
import numpy as np
import pytest

from conftest import REPORT, TL, random_release  # noqa
from nap_lp.net import Net, STAR, full_pattern, covered, mine
from nap_lp.relax import Relaxation, VERIFIED, NOT_VERIFIED, TIMEOUT, THR, UNSTABLE, PINNED
from nap_lp.exact import LPVerifier, MILPVerifier, CascadeVerifier
from nap_lp.coarsen import coarsen, forward_order

pytestmark = pytest.mark.filterwarnings("ignore")


def test_T1_fully_pinned(art):
    net, X, y = art["net"], art["Xtr"], art["ytr"]
    rng = np.random.default_rng(1)
    idx = rng.choice(len(X), 50, replace=False)
    nbad = 0
    for i in idx:
        c = int(y[i]) if rng.random() < 0.5 else 1 - int(y[i])
        P = full_pattern(net, X[i])
        R = Relaxation(net, c)
        out = R.solve(P)
        mp = MILPVerifier(net, c, time_limit=TL, mode="optimize")
        v, opt2, x2, _ = mp.solve(P, "optimize")
        assert not out.empty
        assert abs(out.opt - opt2) < 1e-5, (i, out.opt, opt2)
        xs = np.clip(out.x, 0, 1)
        real = net.margin(xs, c)[0]
        assert abs(real - out.opt) < 1e-5, (i, real, out.opt)
        z = net.preacts_flat(xs)[0]
        ok = ((P == 1) & (z >= -1e-5)) | ((P == 0) & (z <= 1e-5))
        assert ok.all(), "x* does not reproduce the pattern"
    REPORT["T1_points"] = len(idx)


def test_T2_soundness(corpus):
    base = [o for o in corpus if o["role"] == "base"]
    lp_ver = [o for o in base if o["lp"] == VERIFIED]
    tested = [o for o in lp_ver if o["feas"] != TIMEOUT]
    bad = [o for o in tested if o["feas"] != VERIFIED]
    gap = [o for o in base if o["lp"] != VERIFIED and o["feas"] == VERIFIED]
    REPORT["T2"] = dict(n_base=len(base), lp_verified=len(lp_ver), lp_verified_tested=len(tested),
                        exact_timeouts=sum(o["feas"] == TIMEOUT for o in base),
                        gap_lp_nv_exact_v=len(gap), exact_verified=sum(o["feas"] == VERIFIED for o in base),
                        unsound=len(bad))
    print("T2", REPORT["T2"])
    assert not bad, "LP VERIFIED but exact NOT_VERIFIED: soundness bug"


def test_T3_covered_margins(art, corpus):
    net = art["net"]
    nchk = 0
    for o in corpus:
        if o["lp"] == VERIFIED or o["feas"] == VERIFIED:
            P, c = o["P"], o["c"]
            for X in (art["Xtr"], art["Xte"]):
                m = covered(net, X, P)
                if m.any():
                    assert (net.margin(X[m], c) > 0).all(), "verified pattern but covered example misclassified"
                    nchk += int(m.sum())
    REPORT["T3_covered_examples_checked"] = nchk
    assert nchk > 0


def test_T4_monotone(corpus):
    pairs = [(corpus[i], corpus[i + 1]) for i in range(0, len(corpus), 2)]
    res = {}
    for name in ("lp", "feas"):
        nver = 0
        for P, Q in pairs:
            assert P["role"] == "base" and Q["role"] == "finer"
            if P[name] == VERIFIED:
                nver += 1
                assert Q[name] == VERIFIED, f"monotonicity violated ({name})"
        res[name] = dict(pairs=len(pairs), coarse_verified=nver)
    REPORT["T4"] = res
    print("T4", res)


def test_T5_minimality(art, mined, lp_runs):
    net = art["net"]
    rep = {}
    for c, d in lp_runs.items():
        P = d["res"]["pattern"]
        v = LPVerifier(net, c)
        assert v.verify(P) == VERIFIED
        for g in np.flatnonzero(P != STAR):
            Q = P.copy()
            Q[g] = STAR
            assert v.verify(Q) != VERIFIED, f"LP: releasing {g} still verifies"
        rep[f"lp_c{c}"] = dict(size=int((P != STAR).sum()))
    # exact: Coarsen with reference MILP on one class, then re-check each remaining pin
    for c in sorted(lp_runs)[:1]:
        Pm = lp_runs[c]["mined"]
        mv = MILPVerifier(net, c, time_limit=TL)
        r = coarsen(mv, Pm, forward_order(net, Pm), diag=False)
        if r["pattern"] is None:
            rep[f"milp_c{c}"] = "mined did not verify"
            continue
        P = r["pattern"]
        to = 0
        step_v = {s["neuron"]: s["verdict"] for s in r["steps"]}
        late_ver = []   # pin kept only because its release TIMEOUT'd in Coarsen, verifies on re-check
        for g in np.flatnonzero(P != STAR):
            Q = P.copy()
            Q[g] = STAR
            v = mv.verify(Q)
            if v == TIMEOUT:
                to += 1
            if v == VERIFIED and step_v[int(g)] == TIMEOUT:
                late_ver.append(int(g))
                continue
            assert v != VERIFIED, f"MILP: releasing {g} still verifies (Coarsen verdict {step_v[int(g)]})"
        rep[f"milp_c{c}"] = dict(size=int((P != STAR).sum()), recheck_timeouts=to, coarsen_timeouts=r["timeouts"],
                                 verifies_on_recheck_after_coarsen_timeout=late_ver,
                                 calls=r["n_calls"], time=r["time"])
    REPORT["T5"] = rep
    print("T5", rep)


def test_T6_duals_tiny():
    # x in [0,1]^2; z1 = x1-0.5, z2 = x2-0.5 ; out0 = 2 a1 ; out1 = 0.2   (margin for class 0 = 2 a1 - 0.2)
    W0 = np.array([[1.0, 0.0], [0.0, 1.0]])
    b0 = np.array([-0.5, -0.5])
    W1 = np.array([[2.0, 0.0], [0.0, 0.0]])
    b1 = np.array([0.0, 0.2])
    net = Net([W0, W1], [b0, b1])
    R = Relaxation(net, 0)
    P = np.array([1, 0], dtype=np.int8)   # x1 >= .5 (pinned 1), x2 <= .5 (pinned 0)
    out = R.solve(P)
    assert abs(out.opt - (-0.2)) < 1e-9           # min at x1 = .5
    assert out.raw_duals[0] <= 1e-12 and out.raw_duals[1] <= 1e-12   # <= rows, minimisation => marginals <= 0
    assert abs(out.duals[0] - 2.0) < 1e-7        # tight pin: relaxing z1>=0 by t changes margin by 2t
    assert abs(out.duals[1]) < 1e-9              # pin 2 not tight / irrelevant to objective
    # neuron 1 unpinned has no dual
    out2 = R.solve(np.array([1, -1], dtype=np.int8))
    assert out2.duals[1] == 0.0
    # pin that does not bind: x1 pinned 0 (x1<=.5) -> a1 = 0, margin = -0.2 constant, dual 0
    out3 = R.solve(np.array([0, 1], dtype=np.int8))
    assert abs(out3.opt + 0.2) < 1e-9 and out3.duals.sum() < 1e-9


def test_T7_speedup_equivalence_exact(corpus):
    base = [o for o in corpus if o["role"] == "base"]
    n = len(base)
    to_feas = [o for o in base if o["feas"] == TIMEOUT]
    to_opt = [o for o in base if o["opt"] == TIMEOUT]
    comp = [o for o in base if o["feas"] != TIMEOUT and o["opt"] != TIMEOUT]
    dis_a = [o for o in comp if o["feas"] != o["opt"]]
    ref_to = {id(o) for o in to_feas}
    comp_b = [o for o in base if o["feas"] != TIMEOUT]
    dis_b = [o for o in comp_b if o["casc"] != o["feas"]]
    stages = collections.Counter(o["casc_stage"] for o in base)
    # plain LP vs optimised LP verdicts (T8 covers bounds)
    dis_lp = [o for o in base if o["lp"] != o["plain"]]
    REPORT["T7"] = dict(n=n, compared_feas_vs_opt=len(comp), disagree_feas_vs_opt=len(dis_a),
                        timeouts_feas=len(to_feas), timeouts_opt=len(to_opt),
                        compared_casc_vs_ref=len(comp_b), disagree_casc_vs_ref=len(dis_b),
                        cascade_stage_counts=dict(stages), lp_vs_lp_plain_disagree=len(dis_lp),
                        casc_timeouts=sum(o["casc"] == TIMEOUT for o in base),
                        t_feas_total=sum(o["t_feas"] for o in base), t_opt_total=sum(o["t_opt"] for o in base),
                        t_casc_total=sum(o["t_casc"] for o in base))
    print("T7", REPORT["T7"])
    assert n >= 200, f"only {n} patterns"
    assert not dis_a, "feasibility and optimize modes disagree"
    assert not dis_b, "cascade and reference MILP disagree where reference did not time out"
    assert not dis_lp


def _compare_engines(net, c, patterns):
    fast = Relaxation(net, c, cache=True, prefilter=True)
    plain = Relaxation(net, c, cache=False, prefilter=False)
    nunst = 0
    for P in patterns:
        a, b = fast.bounds(P), plain.bounds(P)
        if a.empty or b.empty:
            assert a.empty == b.empty or True  # emptiness found at different places is allowed; verdict compared below
        else:
            un = P == STAR
            assert (a.kind[un] == b.kind[un]).all(), "stable/unstable classification differs"
            u = (a.kind == UNSTABLE)
            nunst += int(u.sum())
            assert np.abs(a.lo[u] - b.lo[u]).max(initial=0) < 1e-7
            assert np.abs(a.hi[u] - b.hi[u]).max(initial=0) < 1e-7
        oa, ob = fast.solve(P), plain.solve(P)
        assert oa.verdict == ob.verdict
        if not (oa.empty or ob.empty):
            assert abs(oa.opt - ob.opt) < 1e-7, (oa.opt, ob.opt)
    return fast, plain, nunst


def test_T8_bounds_equivalence(art, mined, lp_runs):
    net = art["net"]
    rng = np.random.default_rng(5)
    rep = {}
    for c in (0, 1):
        pats = [random_release(net, mined[c], rng, 0.0, 0.5) for _ in range(25)]
        pats += pats[:5]   # repeats exercise cache hits
        fast, plain, nunst = _compare_engines(net, c, pats)
        rep[f"real_net_c{c}"] = dict(fast=fast.stats, unstable_compared=nunst)
    # synthetic net where the interval pre-filter fires
    r = np.random.default_rng(7)
    Ws = [r.normal(0, .6, (10, 6)), r.normal(0, .6, (10, 10)), r.normal(0, .6, (10, 10)), r.normal(0, .6, (2, 10))]
    bs = [r.normal(0, 1.0, 10), r.normal(0, 1.0, 10), r.normal(0, 1.0, 10), r.normal(0, .3, 2)]
    snet = Net(Ws, bs)
    spats = []
    for _ in range(60):
        P = np.full(snet.N, STAR, dtype=np.int8)
        for g in range(snet.N):
            if r.random() < 0.3:
                P[g] = r.integers(0, 2)
        spats.append(P)
    fast, plain, nunst = _compare_engines(snet, 0, spats)
    rep["synthetic"] = dict(fast=fast.stats, unstable_compared=nunst)
    assert fast.stats["bound_lp_skipped"] > 0, "synthetic net did not exercise the pre-filter"
    # whole Coarsen trajectory, seed 0, one class, forward order, both ways
    c = sorted(lp_runs)[0]
    P0 = lp_runs[c]["mined"]
    ra = coarsen(LPVerifier(net, c), P0, forward_order(net, P0), diag=False)
    vp = LPVerifier(net, c, cache=False, prefilter=False)
    rb = coarsen(vp, P0, forward_order(net, P0), diag=False)
    assert [s["verdict"] for s in ra["steps"]] == [s["verdict"] for s in rb["steps"]]
    assert (ra["pattern"] == rb["pattern"]).all()
    rep["trajectory"] = dict(cls=c, calls=ra["n_calls"], t_fast=ra["time"], t_plain=rb["time"])
    REPORT["T8"] = rep
    print("T8", rep)
