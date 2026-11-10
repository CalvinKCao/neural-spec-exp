"""E2: release-order experiments. Results -> results/e2/*.json

usage: python run_e2.py lp  SEED CLASS [--nrand 20]
       python run_e2.py exact SEED CLASS --orders backward dual rand0 rand1 ... [--tl 300] [--procs 8]
Orders: forward, backward, dual (|LP dual| ascending at the mined pattern), randK (rng seed 1000+K).
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nap_lp.data import load
from nap_lp.net import mine, per_layer_counts, STAR
from nap_lp.exact import LPVerifier, MILPVerifier
from nap_lp.coarsen import coarsen, forward_order, backward_order, random_order, dual_order
from run_e1 import coverage

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "e2")


def get_order(net, mined, c, name):
    if name == "forward":
        return forward_order(net, mined), None
    if name == "backward":
        return backward_order(net, mined), None
    if name == "dual":
        o, d = dual_order(net, mined, LPVerifier(net, c).relax)
        return o, (None if d is None else d.tolist())
    if name.startswith("rand"):
        return random_order(net, mined, np.random.default_rng(1000 + int(name[4:]))), None
    raise ValueError(name)


def job(a):
    kind, seed, c, order_name, tl = a
    net, Xtr, ytr, Xte, yte, acc = load(seed)
    mined = mine(net, Xtr, ytr, c, 0.95)
    order, duals = get_order(net, mined, c, order_name)
    V = LPVerifier(net, c) if kind == "lp" else MILPVerifier(net, c, time_limit=tl)
    r = coarsen(V, mined, order, diag=False)
    name = f"seed{seed}_c{c}_{kind}_{order_name}"
    out = dict(name=name, seed=seed, cls=c, kind=kind, order=order_name, tl=tl, mined_verdict=r["mined_verdict"],
               n_calls=r["n_calls"], timeouts=r["timeouts"], wall=r["time"], order_list=order)
    if duals is not None:
        out["duals"] = duals
    if r["pattern"] is not None:
        P = r["pattern"]
        out.update(size=int((P != STAR).sum()), pattern=P.tolist(), layers=per_layer_counts(net, P),
                   cov_train=coverage(net, Xtr, ytr, c, P), cov_test=coverage(net, Xte, yte, c, P),
                   step_verdicts={str(s["neuron"]): s["verdict"] for s in r["steps"]})
    json.dump(out, open(os.path.join(OUT, name + ".json"), "w"))
    print(f"{name}: size={out.get('size')} calls={out['n_calls']} timeouts={out['timeouts']} wall={out['wall']:.1f}s", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["lp", "exact"])
    ap.add_argument("seed", type=int)
    ap.add_argument("cls", type=int)
    ap.add_argument("--nrand", type=int, default=20)
    ap.add_argument("--orders", nargs="*", default=[])
    ap.add_argument("--tl", type=float, default=300)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.kind == "lp":
        names = ["forward", "backward", "dual"] + [f"rand{k}" for k in range(a.nrand)]
        for n in names:
            job(("lp", a.seed, a.cls, n, a.tl))
    else:
        with Pool(a.procs) as p:
            p.map(job, [("exact", a.seed, a.cls, n, a.tl) for n in a.orders], chunksize=1)
