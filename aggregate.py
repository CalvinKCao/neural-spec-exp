"""Build results/*.csv, figures and the tables used in results.md from results/e1, results/e2."""
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
L = [32, 32, 32, 32]


def load_all(sub):
    out = []
    for f in sorted(glob.glob(os.path.join(R, sub, "*.json"))):
        if "marabou" in f:
            continue
        out.append(json.load(open(f)))
    return out


def pins(d):
    return {i for i, v in enumerate(d["pattern"]) if v != -1}


def frac(k, n):
    return f"{k}/{n} ({100 * k / n:.0f}%)"


# ---------- E1 ----------
e1 = load_all("e1")
rows = []
for d in e1:
    row = dict(seed=d["seed"], cls=d["cls"], verifier={"lp": "LP", "lp_plain": "LP plain", "lp_ret2": "LP retighten x2 (variant A)",
                                                      "milp": "MILP reference", "cascade": "cascade (variant)"}[d["kind"]],
               delta=d["delta"], mined_pins=d["mined_size"], mined_verdict=d["mined_verdict"],
               mined_cov_train=frac(d["mined_cov_train"]["own_k"], d["mined_cov_train"]["own_n"]),
               mined_cov_test=frac(d["mined_cov_test"]["own_k"], d["mined_cov_test"]["own_n"]),
               minimal_pins=d.get("size"), layers=d.get("layers"), calls=d["n_calls"], timeouts=d["timeouts"],
               wall_s=round(d["wall"], 1), per_call_s=round(d["mean_call"], 3))
    if "cov_train" in d:
        row.update(cov_train=frac(d["cov_train"]["own_k"], d["cov_train"]["own_n"]),
                   cov_test=frac(d["cov_test"]["own_k"], d["cov_test"]["own_n"]),
                   false_train=frac(d["cov_train"]["false_k"], d["cov_train"]["false_n"]),
                   false_test=frac(d["cov_test"]["false_k"], d["cov_test"]["false_n"]),
                   released=d["released"], released_lp_implied=d["released_implied"])
    rows.append(row)
e1df = pd.DataFrame(rows).sort_values(["seed", "cls", "delta", "verifier"])
e1df.to_csv(os.path.join(R, "e1_main.csv"), index=False)

# overlap LP vs exact, and exact pins split into refuted / unresolved
by = {(d["seed"], d["cls"], d["kind"], d["delta"]): d for d in e1}
ov = []
for (s, c, k, dl), d in by.items():
    if k != "lp" or "pattern" not in d:
        continue
    m = by.get((s, c, "milp", dl))
    if m is None or "pattern" not in m:
        continue
    A, B = pins(d), pins(m)
    refuted = sum(1 for g in B if next((x["verdict"] for x in m["steps"] if x["neuron"] == g), None) == "NOT_VERIFIED")
    ov.append(dict(seed=s, cls=c, lp_pins=len(A), exact_pins=len(B), lp_minus_exact=len(A - B), exact_minus_lp=len(B - A),
                   jaccard=round(len(A & B) / len(A | B), 3) if A | B else None,
                   exact_pins_refuted_by_counterexample=refuted, exact_pins_timeout_only=len(B) - refuted,
                   lp_calls=d["n_calls"], lp_wall=round(d["wall"], 1), exact_calls=m["n_calls"], exact_wall=round(m["wall"], 1),
                   exact_timeouts=m["timeouts"]))
ovdf = pd.DataFrame(ov)
ovdf.to_csv(os.path.join(R, "e1_lp_vs_exact.csv"), index=False)

# mined baseline: does it verify under each verifier?
base = []
for s in range(5):
    for c in (0, 1):
        lp, ex = by.get((s, c, "lp", 0.95)), by.get((s, c, "milp", 0.95))
        base.append(dict(seed=s, cls=c, mined_pins=lp["mined_size"] if lp else None,
                         lp=lp["mined_verdict"] if lp else "-", exact_milp=ex["mined_verdict"] if ex else "(not finished)",
                         lp_retighten_x2=(by.get((s, c, "lp_ret2", 0.95)) or {}).get("mined_verdict", "-"),
                         lp_delta99=(by.get((s, c, "lp", 0.99)) or {}).get("mined_verdict", "-"),
                         milp_delta99=(by.get((s, c, "milp", 0.99)) or {}).get("mined_verdict", "-")))
basedf = pd.DataFrame(base)
basedf.to_csv(os.path.join(R, "e1_baseline.csv"), index=False)

# per-layer figure
fig, ax = plt.subplots(figsize=(6, 3.4))
tot = {"LP": np.zeros(4), "exact (MILP)": np.zeros(4)}
for (s, c, k, dl), d in by.items():
    if dl != 0.95 or "layers" not in d:
        continue
    if k == "lp":
        tot["LP"] += d["layers"]
    if k == "milp":
        tot["exact (MILP)"] += d["layers"]
w = 0.38
for i, (nm, v) in enumerate(tot.items()):
    ax.bar(np.arange(4) + (i - .5) * w, v, w, label=nm)
ax.set_xticks(range(4), [f"layer {i + 1}" for i in range(4)])
ax.set_ylabel("pinned neurons (summed over cases)")
ax.legend(frameon=False)
ax.set_title("Where the minimal NAPs keep their pins")
fig.tight_layout()
fig.savefig(os.path.join(R, "fig_pins_per_layer.png"), dpi=150)
plt.close(fig)

# ---------- E2 ----------
e2 = load_all("e2")
E2 = {}
for d in e2:
    E2.setdefault((d["seed"], d["cls"], d["kind"]), {})[d["order"]] = d
rows, ess = [], []
for (s, c, k), od in sorted(E2.items()):
    ok = {o: d for o, d in od.items() if "pattern" in d}
    rnd = [d for o, d in ok.items() if o.startswith("rand")]
    row = dict(seed=s, cls=c, verifier=k, n_orders=len(ok), n_random=len(rnd))
    for o in ("forward", "backward", "dual"):
        if o in ok:
            d = ok[o]
            row.update({f"{o}_pins": d["size"], f"{o}_calls": d["n_calls"], f"{o}_test": frac(d["cov_test"]["own_k"], d["cov_test"]["own_n"]),
                        f"{o}_timeouts": d["timeouts"]})
    if rnd:
        sz = [d["size"] for d in rnd]
        ct = [d["cov_test"]["own_k"] for d in rnd]
        cl = [d["n_calls"] for d in rnd]
        row.update(rand_pins_mean=round(np.mean(sz), 1), rand_pins_min=min(sz), rand_pins_max=max(sz),
                   rand_calls_mean=round(np.mean(cl), 1), rand_test_k_min=min(ct), rand_test_k_max=max(ct),
                   test_n=rnd[0]["cov_test"]["own_n"])
    rows.append(row)
e2df = pd.DataFrame(rows)
e2df.to_csv(os.path.join(R, "e2_orders.csv"), index=False)

# union / intersection
for (s, c) in sorted({(s, c) for (s, c, k) in E2}):
    lp = [pins(d) for d in E2.get((s, c, "lp"), {}).values() if "pattern" in d]
    ex_all = [d for d in E2.get((s, c, "exact"), {}).values() if "pattern" in d]
    if not lp:
        continue
    lpU, lpI = set.union(*lp), set.intersection(*lp)
    rec = dict(seed=s, cls=c, n_lp_orders=len(lp), lp_union=len(lpU), lp_intersection=len(lpI))
    if ex_all:
        # seed-0 forward exact run comes from E1
        exs = [pins(d) for d in ex_all]
        f = by.get((s, c, "milp", 0.95))
        if f and "pattern" in f and "forward" not in E2[(s, c, "exact")]:
            exs.append(pins(f))
            ex_all = ex_all + [dict(pattern=f["pattern"], step_verdicts={str(x["neuron"]): x["verdict"] for x in f["steps"]})]
        exU, exI = set.union(*exs), set.intersection(*exs)
        # strict: pinned in every order AND refuted by a counterexample (not a timeout) in every order
        strictI = {g for g in exI if all(d["step_verdicts"].get(str(g)) == "NOT_VERIFIED" for d in ex_all)}
        rec.update(n_exact_orders=len(exs), exact_union=len(exU), exact_intersection=len(exI),
                   exact_intersection_refuted_every_time=len(strictI),
                   relaxation_essential=sorted(lpI - exU), robustly_essential=sorted(lpI & exI),
                   robustly_essential_strict=sorted(lpI & strictI), lp_intersection_set=sorted(lpI),
                   exact_union_set=sorted(exU), exact_intersection_set=sorted(exI))
    ess.append(rec)
pd.DataFrame(ess).to_csv(os.path.join(R, "e2_essential.csv"), index=False)

# coverage vs size figure for the LP random orders (one panel per case)
cases = sorted({(s, c) for (s, c, k) in E2 if k == "lp"})
if cases:
    fig, axs = plt.subplots(1, len(cases), figsize=(3.2 * len(cases), 3.2), squeeze=False)
    for ax, (s, c) in zip(axs[0], cases):
        od = E2[(s, c, "lp")]
        rnd = [d for o, d in od.items() if o.startswith("rand") and "pattern" in d]
        ax.scatter([d["size"] for d in rnd], [d["cov_test"]["own_k"] for d in rnd], s=14, c="0.6", label="random")
        for o, mk, col in (("forward", "o", "C0"), ("backward", "s", "C1"), ("dual", "^", "C2")):
            if o in od and "pattern" in od[o]:
                ax.scatter([od[o]["size"]], [od[o]["cov_test"]["own_k"]], marker=mk, c=col, s=50, label=o, zorder=3)
        n = rnd[0]["cov_test"]["own_n"] if rnd else ""
        ax.set_title(f"seed {s}, class {c}", fontsize=9)
        ax.set_xlabel("pinned neurons")
        ax.set_ylabel(f"test examples covered (of {n})")
    axs[0][0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(R, "fig_coverage_vs_size.png"), dpi=150)
    plt.close(fig)

# ---------- timing table (seed 0 class 1) ----------
tim = []
for k, nm in (("milp", "MILP reference"), ("cascade", "cascade (variant)"), ("lp", "LP"), ("lp_plain", "LP plain")):
    d = by.get((0, 1, k, 0.95))
    if d:
        r = d.get("relax_stats", {})
        tim.append(dict(verifier=nm, calls=d["n_calls"], total_s=round(d["wall"], 1), per_call_s=round(d["mean_call"], 3),
                        timeouts=d["timeouts"], bound_lps_run=r.get("bound_lp"), bound_lps_skipped_by_prefilter=r.get("bound_lp_skipped"),
                        cache_hit_rate=(round(r["cache_hits"] / r["cache_lookups"], 2) if r.get("cache_lookups") else None),
                        cascade_stages=d.get("stage_counts")))
pd.DataFrame(tim).to_csv(os.path.join(R, "e1_timing_seed0_c1.csv"), index=False)

for name, df in (("baseline", basedf), ("overlap", ovdf), ("timing", pd.DataFrame(tim)), ("e2", e2df), ("essential", pd.DataFrame(ess))):
    print("\n==", name)
    print(df.to_string(index=False) if len(df) else "(empty)")
print("\n== E1 main")
print(e1df.drop(columns=["layers"]).to_string(index=False))
