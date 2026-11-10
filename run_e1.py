"""E1: Coarsen with each verifier. One job = (seed, class, kind, delta, order). Results -> results/e1/*.json

kinds: lp (cache+prefilter), lp_plain (neither), lp_ret2 (variant A: 2-round re-tightening),
       milp (REFERENCE exact, feasibility mode), cascade (variant)
usage: python run_e1.py SEED [--tl 300] [--tag label] [--procs 6]
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
from nap_lp.net import mine, covered, per_layer_counts, STAR
from nap_lp.exact import LPVerifier, MILPVerifier, CascadeVerifier
from nap_lp.coarsen import coarsen, forward_order
from nap_lp.relax import VERIFIED, TIMEOUT

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "e1")


def coverage(net, X, y, c, P):
    m = covered(net, X, P)
    own, oth = y == c, y != c
    return dict(own_k=int(m[own].sum()), own_n=int(own.sum()), false_k=int(m[oth].sum()), false_n=int(oth.sum()))


def make_verifier(kind, net, c, tl):
    if kind == "lp":
        return LPVerifier(net, c)
    if kind == "lp_plain":
        return LPVerifier(net, c, cache=False, prefilter=False)
    if kind == "lp_ret2":
        return LPVerifier(net, c, retighten=2)
    if kind == "milp":
        return MILPVerifier(net, c, time_limit=tl, mode="feasibility")
    if kind == "cascade":
        return CascadeVerifier(net, c, time_limit=tl)
    raise ValueError(kind)


def job(a):
    seed, c, kind, delta, tl, tag = a
    net, Xtr, ytr, Xte, yte, acc = load(seed)
    mined = mine(net, Xtr, ytr, c, delta)
    name = f"seed{seed}_c{c}_{kind}_d{delta}"
    V = make_verifier(kind, net, c, tl)
    t0 = time.perf_counter()
    r = coarsen(V, mined, forward_order(net, mined))
    wall = time.perf_counter() - t0
    out = dict(name=name, seed=seed, cls=c, kind=kind, delta=delta, tl=tl, tag=tag, test_acc=acc,
               mined_size=int((mined != STAR).sum()), mined_layers=per_layer_counts(net, mined),
               mined_verdict=r["mined_verdict"], mined_cov_train=coverage(net, Xtr, ytr, c, mined),
               mined_cov_test=coverage(net, Xte, yte, c, mined), wall=wall, n_calls=r["n_calls"],
               timeouts=r["timeouts"], mean_call=wall / max(r["n_calls"], 1))
    if r["pattern"] is not None:
        P = r["pattern"]
        st = r["steps"]
        out.update(size=int((P != STAR).sum()), layers=per_layer_counts(net, P), pattern=P.tolist(),
                   cov_train=coverage(net, Xtr, ytr, c, P), cov_test=coverage(net, Xte, yte, c, P),
                   released=sum(s["released"] for s in st),
                   released_implied=sum(bool(s.get("implied")) for s in st if s["released"]),
                   steps=[{k: v for k, v in s.items() if k != "tried"} for s in st])
        np.save(os.path.join(OUT, name + "_tried.npy"), np.array([s["tried"] for s in st], dtype=np.int8))
    out["call_log"] = [{k: (None if (isinstance(v, float) and v != v) else v) for k, v in e.items()} for e in r["call_log"]]
    if hasattr(V, "relax"):
        out["relax_stats"] = V.relax.stats
    if kind == "cascade":
        out["stage_counts"] = {s: sum(e["stage"] == s for e in r["call_log"]) for s in ("lp", "lp_cex", "milp")}
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(out, f, default=float)
    print(f"[{tag}] {name}: mined={out['mined_verdict']} size={out.get('size')} calls={out['n_calls']} "
          f"timeouts={out['timeouts']} wall={wall:.1f}s", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seeds", type=int, nargs="+")
    ap.add_argument("--tl", type=float, default=300)
    ap.add_argument("--tag", default="final")
    ap.add_argument("--procs", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tl, tag = args.tl, args.tag
    # phase A: fast LP runs decide whether the pre-declared fallback is needed.
    # seed 0 also gets plain LP + cascade; other seeds: LP + reference MILP only.
    fast, slow = [], []
    for s in args.seeds:
        for c in (0, 1):
            fast.append((s, c, "lp", 0.95, tl, tag))
            slow.append((s, c, "milp", 0.95, tl, tag))
            if s == 0:
                fast.append((s, c, "lp_plain", 0.95, tl, tag))
                slow.append((s, c, "cascade", 0.95, tl, tag))
    res = [job(a) for a in fast]
    for o in res:
        if o["kind"] == "lp" and o["mined_verdict"] != VERIFIED:
            s, c = o["seed"], o["cls"]
            print(f"[fallback] seed {s} class {c}: mined NAP does not verify under LP -> variants A (retighten) and B (delta 0.99)", flush=True)
            slow += [(s, c, "lp_ret2", 0.95, tl, tag + "-variantA"),
                     (s, c, "lp", 0.99, tl, tag + "-variantB"), (s, c, "milp", 0.99, tl, tag + "-variantB")]
    slow.sort(key=lambda a: a[2] != "milp")   # reference MILP runs (the slow ones) first
    with Pool(args.procs) as p:
        p.map(job, slow, chunksize=1)
    print("E1 done")
