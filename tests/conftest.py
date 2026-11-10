import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")

from nap_lp.data import load  # noqa: E402
from nap_lp.net import mine, STAR  # noqa: E402
from nap_lp.exact import LPVerifier, MILPVerifier, CascadeVerifier  # noqa: E402
from nap_lp.coarsen import coarsen, forward_order  # noqa: E402
from nap_lp.relax import VERIFIED, TIMEOUT  # noqa: E402

TL = float(os.environ.get("NAP_MILP_TL", "60"))          # per-call MILP limit in tests
N_RAND = int(os.environ.get("NAP_TEST_NRAND", "80"))      # random-release patterns per class
NPROC = int(os.environ.get("NAP_TEST_PROCS", "6"))
SEED = 0

REPORT = {}


def pytest_sessionfinish(session, exitstatus):
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "test_report.json"), "w") as f:
        json.dump(REPORT, f, indent=1, default=str)


@pytest.fixture(scope="session")
def art():
    net, Xtr, ytr, Xte, yte, acc = load(SEED)
    return dict(net=net, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, acc=acc)


def random_release(net, mined, rng, lo=0.0, hi=0.3):
    P = mined.copy()
    pins = np.flatnonzero(P != STAR)
    k = int(round(rng.uniform(lo, hi) * len(pins)))
    P[rng.choice(pins, size=k, replace=False)] = STAR
    return P


def finer_partner(mined, P, rng):
    """P' pins a random subset of the neurons that are pinned in mined but released in P, at mined values."""
    Q = P.copy()
    rel = np.flatnonzero((mined != STAR) & (P == STAR))
    if len(rel):
        k = rng.integers(1, len(rel) + 1)
        sel = rng.choice(rel, size=k, replace=False)
        Q[sel] = mined[sel]
    return Q


_W = {}


def _init(seed, tl):
    net, *_ = load(seed)
    _W["net"], _W["tl"] = net, tl


def _eval(job):
    """Evaluate one pattern with every verifier. job = (c, pattern, do_full)."""
    c, P, full = job
    net, tl = _W["net"], _W["tl"]
    key = ("v", c)
    if key not in _W:
        _W[key] = dict(lp=LPVerifier(net, c), plain=LPVerifier(net, c, cache=False, prefilter=False),
                       feas=MILPVerifier(net, c, time_limit=tl, mode="feasibility"),
                       opt=MILPVerifier(net, c, time_limit=tl, mode="optimize"),
                       casc=CascadeVerifier(net, c, time_limit=tl))
    V = _W[key]
    out = dict(c=c, P=P)
    out["lp"] = V["lp"].verify(P)
    out["lp_opt"] = V["lp"].log[-1]["opt"]
    t = time.perf_counter()
    out["feas"] = V["feas"].verify(P)
    out["t_feas"] = time.perf_counter() - t
    if full:
        out["plain"] = V["plain"].verify(P)
        t = time.perf_counter()
        out["opt"] = V["opt"].verify(P)
        out["opt_val"] = V["opt"].log[-1]["opt"]
        out["t_opt"] = time.perf_counter() - t
        t = time.perf_counter()
        out["casc"] = V["casc"].verify(P)
        out["casc_stage"] = V["casc"].log[-1]["stage"]
        out["t_casc"] = time.perf_counter() - t
    return out


@pytest.fixture(scope="session")
def mined(art):
    return {c: mine(art["net"], art["Xtr"], art["ytr"], c, 0.95) for c in (0, 1)}


@pytest.fixture(scope="session")
def lp_runs(art, mined):
    """LP Coarsen (forward order) for each class whose mined pattern verifies; used for trajectories (T2/T4/T5/T8)."""
    net = art["net"]
    runs = {}
    for c in (0, 1):
        for delta in (0.95, 0.99):
            P = mine(net, art["Xtr"], art["ytr"], c, delta)
            v = LPVerifier(net, c)
            r = coarsen(v, P, forward_order(net, P))
            if r["pattern"] is not None:
                runs[c] = dict(delta=delta, mined=P, res=r)
                break
    REPORT["lp_coarsen_runs"] = {c: dict(delta=d["delta"], size=int((d["res"]["pattern"] != STAR).sum()),
                                         calls=d["res"]["n_calls"]) for c, d in runs.items()}
    return runs


@pytest.fixture(scope="session")
def corpus(art, mined, lp_runs):
    """Non-vacuous pattern corpus: (a) random releases 0-30% of mined pins, (b) every pattern tried in LP Coarsen
    trajectories. Each pattern gets a finer partner (for T4). Evaluated with all verifiers in parallel."""
    net = art["net"]
    rng = np.random.default_rng(123)
    jobs, meta = [], []
    for c in (0, 1):
        base = []
        for _ in range(N_RAND):
            base.append(("rand", random_release(net, mined[c], rng)))
        if c in lp_runs:
            for s in lp_runs[c]["res"]["steps"]:
                base.append(("traj", s["tried"]))
        mref = lp_runs[c]["mined"] if c in lp_runs else mined[c]
        for src, P in base:
            jobs.append((c, P, True))
            meta.append(dict(src=src, role="base"))
            Q = finer_partner(mref, P, rng)
            jobs.append((c, Q, False))
            meta.append(dict(src=src, role="finer"))
    t = time.perf_counter()
    with Pool(NPROC, initializer=_init, initargs=(SEED, TL)) as pool:
        outs = pool.map(_eval, jobs, chunksize=1)
    REPORT["corpus_time_s"] = time.perf_counter() - t
    for o, m in zip(outs, meta):
        o.update(m)
    REPORT["corpus_size"] = len(outs)
    return outs
