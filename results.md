# Results

All numbers come from `results/*.csv` (built by `aggregate.py` from `results/e1/*.json`, `results/e2/*.json`).
Property: margin > 1e-6 over the NAP region inside [0,1]^30. Exact MILP: 5-minute limit per call, a timeout counts
as NOT_VERIFIED (pin stays). Forward release order unless stated. Machine: 8 cores, MILP/E2 jobs overlapped
(load average 11 to 19), so absolute times for seeds 1-4 and E2 are inflated. Seed 0 E1 ran with 6 workers.

Verifier modes: **MILP ref** = reference exact verifier (HiGHS, feasibility mode); **cascade** = variant
(LP, then LP counterexample check, then MILP); **LP** = relaxation with cache + interval pre-filter;
**LP plain** = same without them. Seeds 1-4 used MILP ref only (no cascade, no plain LP).

## Tests (local run, seed 0, MILP limit 60 s): 8 passed

| Test | Result |
|---|---|
| T1 | LP optimum = exact optimum, x* reproduces margin, 50 fully pinned points |
| T2 | 269 patterns; 141 LP-VERIFIED and tested, 0 unsound; 189 exact-VERIFIED; gap (LP no, exact yes) 48; 25 exact timeouts excluded |
| T3 | all covered train/test examples of verified patterns have margin > 0 |
| T4 | 269 pairs, 141 (LP) and 189 (exact) coarse patterns verified, no violation |
| T5 | LP minimal for class 1 (8 pins). MILP Coarsen left 6 pins (6 timeouts); pin 123 verifies on re-check after a Coarsen timeout, listed not failed |
| T6 | duals non-negative where expected, zero for slack pins (tiny net) |
| T7 | MILP feasibility vs optimize: 131 comparable, 0 disagreements (optimize timed out on 136). Cascade vs reference: 244 comparable, 0 disagreements. Cascade stages: LP 141, LP counterexample 0, MILP 128 |
| T8 | cache + pre-filter: identical classification, bounds within 1e-7, same margins, identical Coarsen trajectory (110 calls) |

The pre-filter skipped 0 bound LPs on the real networks (it fires only on a synthetic test network). The LP-counterexample
stage never fired: the LP minimizer sits on the pin boundary and fails the no-tolerance check.
Marabou is not part of pytest; see below.

## E1, mined baseline (delta 0.95): does the mined NAP verify?

`results/e1_baseline.csv`. Pre-declared fallbacks (A: LP with 2-round re-tightening, B: delta 0.99) were run wherever LP failed.

| seed | class | mined pins | LP | MILP ref | A (LP retighten x2) | B (delta 0.99, LP / MILP) |
|---|---|---|---|---|---|---|
| 0 | 0 | 80 | no | no | no | no / no |
| 0 | 1 | 109 | yes | yes | | |
| 1 | 0 | 101 | yes | yes | | |
| 1 | 1 | 82 | no | no | no | no / no |
| 2 | 0 | 100 | yes | yes | | |
| 2 | 1 | 88 | no | run stopped after 2 h, not finished | no | no / no |
| 3 | 0 | 80 | no | no | no | no / no |
| 3 | 1 | 98 | yes | yes | | |
| 4 | 0 | 98 | yes | yes | | |
| 4 | 1 | 90 | no | yes | yes (14 pins, 2468 s) | no / no |

LP verifies 5 of 10, MILP ref 6 of the 9 it finished. Seed 4 class 1 is the one case where the exact verifier proves the
mined NAP and plain LP does not (LP with re-tightening does). Variant B never helped.

## E1, Coarsen: LP vs exact (`results/e1_lp_vs_exact.csv`)

| seed, class | LP pins | MILP ref pins | LP minus exact | exact minus LP | Jaccard | LP calls / time / per call | MILP calls / time / per call | MILP timeouts |
|---|---|---|---|---|---|---|---|---|
| 0, 1 | 8 | 3 | 6 | 1 | 0.22 | 110 / 7.6 s / 0.07 s | 110 / 1933 s / 17.6 s | 3 |
| 1, 0 | 11 | 5 | 10 | 4 | 0.07 | 102 / 18 s / 0.18 s | 102 / 3958 s / 38.8 s | 5 |
| 2, 0 | 15 | 8 | 11 | 4 | 0.21 | 101 / 22 s / 0.22 s | 101 / 3813 s / 37.8 s | 8 |
| 3, 1 | 9 | 6 | 6 | 3 | 0.25 | 99 / 27 s / 0.27 s | 99 / 4411 s / 44.6 s | 6 |
| 4, 0 | 30 | 3 | 27 | 0 | 0.10 | 99 / 80 s / 0.81 s | 99 / 3476 s / 35.1 s | 3 |
| 4, 1 (LP = variant A) | 14 | 21 | | | | 91 / 2468 s / 27.1 s | 91 / 7151 s / 78.6 s | 21 |

Important: in every row the number of exact pins equals the number of MILP timeouts. All exact-pinned neurons are neurons whose
release timed out, none was refuted by a counterexample (`exact_pins_refuted_by_counterexample` = 0 in the csv).
The exact sizes are therefore upper bounds from the 5-minute limit, and the LP-minus-exact and exact-minus-LP counts mostly
compare "LP could not prove release" with "MILP did not finish". In the seed 4 class 1 row, LP (variant A) is slower per call than usual because of the re-tightening LPs, and its set
is smaller than the exact one only because the exact one is cut off by timeouts.

All pinned neurons of all minimal sets lie in layer 4 (`results/fig_pins_per_layer.png`).

Coverage (own-class, k/n; false coverage was 0 to 2 other-class examples everywhere, see `e1_main.csv`):

| seed, class | train: mined / LP / exact | test: mined / LP / exact |
|---|---|---|
| 0, 1 | 258/285, 281/285, 282/285 | 61/72, 69/72, 69/72 |
| 1, 0 | 154/170, 166/170, 166/170 | 37/42, 40/42, 40/42 |
| 2, 0 | 157/170, 164/170, 164/170 | 39/42, 39/42, 39/42 |
| 3, 1 | 242/285, 271/285, 271/285 | 58/72, 67/72, 67/72 |
| 4, 0 | 151/170, 160/170, 164/170 | 35/42, 37/42, 39/42 |

Test accuracy of the networks (seeds 0-4): 0.965, 0.974, 0.974, 0.982, 0.965.

LP-implied releases (diagnostic, lower bound on true implication): 1 of 76 successful releases in the only case where it was
nonzero (seed 4 class 1, variant A); 0 in all other LP runs and in all exact runs. Almost every release enlarged the region.

## Timing, seed 0 class 1 (`results/e1_timing_seed0_c1.csv`)

| verifier | calls | total | per call | timeouts | bound LPs run | skipped by pre-filter | cache hit rate |
|---|---|---|---|---|---|---|---|
| MILP ref | 110 | 1933 s | 17.6 s | 3 | 1044 | 0 | 0.82 |
| cascade (variant) | 110 | 1934 s | 17.6 s | 3 | 1044 | 0 | 0.84 (stages: LP 81, LP counterexample 0, MILP 29) |
| LP | 110 | 7.6 s | 0.069 s | 0 | 1044 | 0 | 0.82 |
| LP plain | 110 | 103.5 s | 0.94 s | 0 | 30110 | n/a | 0 |

Caching gives about 13.6x on the LP run; the pre-filter contributes nothing on this network. The cascade saved no time; both
exact runs gave the same final pattern. Seeds 1-4 used MILP ref only, so no cascade rows exist for them.

## Marabou cross-check (seed 0 class 1, MILP ref trajectory)

109 queries from the reference MILP Coarsen run, 5-minute limit, Marabou (maraboupy 2.0.0, Python 3.11 env).
0 disagreements with the MILP on queries both decided; 28 Marabou timeouts (MILP: 3); every SAT counterexample was valid
on the real network. Details: `results/e1/seed0_c1_milp_d0.95_marabou.json`. Marabou was slower than the MILP here.
Cross-check queries for other seeds were not run.

## E2, release order (`results/e2_orders.csv`, `results/e2_essential.csv`, `results/fig_coverage_vs_size.png`)

LP, pinned neurons in the minimal NAP (forward / backward / dual-guided / random mean, min-max). Seed 4 class 0 used 6 random orders, the others 20.

| seed, class | forward | backward | dual | random mean (min-max) | test coverage fwd / bwd / dual / random range |
|---|---|---|---|---|---|
| 0, 1 | 8 | 63 | 8 | 52.2 (8-78) | 69 / 61 / 69 / 61-69 of 72 |
| 1, 0 | 11 | 69 | 11 | 51.0 (21-71) | 40 / 37 / 40 / 37-39 of 42 |
| 2, 0 | 15 | 70 | 15 | 58.2 (33-72) | 39 / 39 / 39 / 39 of 42 |
| 3, 1 | 9 | 55 | 9 | 53.1 (9-74) | 67 / 58 / 67 / 58-67 of 72 |
| 4, 0 | 30 | 73 | 30 | 62.3 (55-71) | 37 / 35 / 37 / 35-37 of 42 |

Order changes the LP result a lot, and forward order was the smallest in every case. Verifier calls were about the same in every order
(about 100) since each order makes one call per mined pin. Dual-guided order differs from forward order only in releasing the 3 to 8 pins with nonzero dual
last, and gave the identical final set as forward order in all five cases, so it did nothing beyond forward order here.

Exact (seed 0 class 1 only; only 3 orders finished): forward 3 pins (3 timeouts), dual-guided 4 pins (4 timeouts),
random order 2: 8 pins (8 timeouts, 6160 s). Backward and 5 more random orders were stopped after about 2 hours without finishing.

Essential-neuron estimate (seed 0 class 1): LP intersection over 23 orders = {115}; LP union = 92 neurons. Exact union over the
3 finished orders = 12 neurons and includes 115; exact intersection is empty. So relaxation-essential = empty and robustly
essential = empty. This is not a finding about the paper's set E: only 3 exact orders finished and every exact pin was a timeout.
For the other four cases only LP orders were run: LP intersection sizes 0, 0, 2, 2 (unions 85, 89, 82, 83 neurons).

## Not done or incomplete

- Seed 2 class 1: MILP reference at delta 0.95 was stopped after more than 2 hours; no result.
- Exact E2: 3 of 8 planned orders finished, seed 0 class 1 only.
- Cascade, plain LP and Marabou: seed 0 only (Marabou on class 1 only).
- E3 (delta sweep): not run (delta 0.99 appears only as fallback variant B).
- Neither verifier's timeouts were chased with longer limits.
