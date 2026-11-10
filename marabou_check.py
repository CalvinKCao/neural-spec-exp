"""Marabou cross-check (run in .venv-marabou, Python 3.11; needs only numpy + maraboupy + nap_lp.net/data).

Query: x in [0,1]^30, pins as bounds on neuron pre-activations (pin1: z >= 0, pin0: z <= 0), every hidden neuron
has a ReLU constraint, and the negated property margin <= 1e-6. UNSAT -> VERIFIED, SAT -> NOT_VERIFIED.

usage: python marabou_check.py RUN_NAME [--tl 300] [--procs 2] [--limit N]
RUN_NAME e.g. seed0_c1_milp_d0.95 : uses results/e1/RUN_tried.npy as the queries and the MILP verdicts
stored in results/e1/RUN.json (call_log[1:] = steps). Writes results/e1/RUN_marabou.json.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from maraboupy import MarabouCore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nap_lp.data import load

THR = 1e-6
E1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "e1")


def build_query(net, c, pattern):
    iq = MarabouCore.InputQuery()
    nv = net.n_in + 2 * net.N + 1
    iq.setNumberOfVariables(nv)
    xi = list(range(net.n_in))
    for v in xi:
        iq.setLowerBound(v, 0.0)
        iq.setUpperBound(v, 1.0)
        iq.markInputVariable(v, v)
    prev = xi
    nxt = net.n_in
    for l in range(net.L):
        w = net.widths[l]
        z = list(range(nxt, nxt + w))
        a = list(range(nxt + w, nxt + 2 * w))
        nxt += 2 * w
        for j in range(w):
            eq = MarabouCore.Equation(MarabouCore.Equation.EQ)
            eq.addAddend(1.0, z[j])
            for k, p in enumerate(prev):
                if net.W[l][j, k] != 0.0:
                    eq.addAddend(-float(net.W[l][j, k]), p)
            eq.setScalar(float(net.b[l][j]))
            iq.addEquation(eq)
            g = net.off[l] + j
            if pattern[g] == 1:
                iq.setLowerBound(z[j], 0.0)
            elif pattern[g] == 0:
                iq.setUpperBound(z[j], 0.0)
            MarabouCore.addReluConstraint(iq, z[j], a[j])
        prev = a
    m = nxt
    wm, bm = net.margin_vec(c)
    eq = MarabouCore.Equation(MarabouCore.Equation.EQ)
    eq.addAddend(1.0, m)
    for k, p in enumerate(prev):
        eq.addAddend(-float(wm[k]), p)
    eq.setScalar(bm)
    iq.addEquation(eq)
    iq.setUpperBound(m, THR)     # negated property: margin <= THR
    iq.markOutputVariable(m, 0)
    return iq


_S = {}


def solve_one(a):
    seed, c, pattern, tl = a
    if "net" not in _S:
        _S["net"] = load(seed)[0]
    net = _S["net"]
    iq = build_query(net, c, pattern)
    opt = MarabouCore.Options()
    opt._timeoutInSeconds = int(tl)
    opt._verbosity = 0
    t = time.perf_counter()
    code, vals, _ = MarabouCore.solve(iq, opt, "")
    dt = time.perf_counter() - t
    v = {"unsat": "VERIFIED", "sat": "NOT_VERIFIED", "TIMEOUT": "TIMEOUT"}.get(code, code)
    cex_ok = None
    if code == "sat":   # check the counterexample on the real network
        x = np.array([vals[i] for i in range(net.n_in)])
        zs, lg = net.forward(np.clip(x, 0, 1))
        z = np.concatenate([zz[0] for zz in zs])
        cex_ok = bool((z[pattern == 1] >= -1e-6).all() and (z[pattern == 0] <= 1e-6).all()
                      and lg[0, c] - lg[0, 1 - c] <= THR + 1e-5)
    return dict(verdict=v, time=dt, cex_ok=cex_ok)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--tl", type=float, default=300)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    meta = json.load(open(os.path.join(E1, args.run + ".json")))
    tried = np.load(os.path.join(E1, args.run + "_tried.npy"))
    ref = [e["verdict"] for e in meta["call_log"][1:]]
    n = args.limit or len(tried)
    jobs = [(meta["seed"], meta["cls"], tried[i], args.tl) for i in range(n)]
    with Pool(args.procs) as p:
        outs = p.map(solve_one, jobs, chunksize=1)
    rows, dis = [], []
    for i, o in enumerate(outs):
        o["milp"] = ref[i]
        o["i"] = i
        rows.append(o)
        if "TIMEOUT" not in (o["verdict"], o["milp"]) and o["verdict"] != o["milp"]:
            dis.append(i)
    summ = dict(run=args.run, n=n, disagreements=dis, marabou_timeouts=sum(o["verdict"] == "TIMEOUT" for o in outs),
                milp_timeouts=sum(r == "TIMEOUT" for r in ref[:n]), bad_cex=[o["i"] for o in outs if o["cex_ok"] is False],
                marabou_time=sum(o["time"] for o in outs), rows=rows)
    json.dump(summ, open(os.path.join(E1, args.run + "_marabou.json"), "w"))
    print({k: v for k, v in summ.items() if k != "rows"})
    if dis:
        print("DISAGREEMENT between Marabou and MILP on queries", dis)
        sys.exit(2)
