# CLAUDE.md — Project Brief for Claude Code

This file is read automatically by Claude Code at session start. Keep it
concise: it taxes every turn's context window. For background reading
that does **not** belong here, see `README.md`.

---

## What this repo is

Python simulation backing an **IEEE Transactions manuscript** on
holographic-ELAA-enabled near-field ISAC. Two algorithmic deliverables:

- The **system model** (Sec. II–III of the manuscript): near-field array
  response, RHS phase matrix Φ, CU channels, ST/CU/EO target responses,
  sum-rate, and the asymptotic sensing mutual information (Prop. 1).
- The **beamforming algorithms** (Sec. IV–V):
  - **SOOP1** — communication-centric (ZF + water-filling + accelerated
    non-homogeneous quadratic transform on the amplitude `a`).
  - **SOOP2** — sensing-centric (CVX-based SDP for the digital cov.
    matrix Ω, EVD factorisation, projected gradient on `a`).
  - **MOOP** — Tchebycheff scalarisation, FP for sum-rate surrogate,
    first-order convex inequality for sensing surrogate, **Lagrangian
    dual** loop on `(a, λ₁, λ₂)`.

The manuscript itself is **not** committed to the repo (IP-sensitive).
If you need a passage, ask the user to paste only that passage.

---

## Repository map

```
config.py               SystemConfig + AlgorithmConfig dataclasses.
                        Single source of truth for parameters.

system_model/           Pure scenario generation. NO optimisation here.
  geometry.py           Element / feed positions (RHS on y-z plane).
  rhs.py                Φ (Eq. 1), F = diag(a) Φ.
                        Two RHS structures: "fully_connected", "subarray".
  channel.py            Near-field array response (Eq. 13–19),
                        CU channel h_k (Eq. 11),
                        target response G[s] (Eq. 12).
  scenario.py           Scenario dataclass — single object the algorithms
                        consume. ANY new field needed by an algorithm
                        gets added here so algorithms stay model-agnostic.

algorithms/             Pure optimisation. Each function takes a
                        Scenario + cfg and returns {W, A, a, F, history}.
  utils.py              Pure metric functions (sum_rate, sensing_mi,
                        compute_F, transmit_power, scale_W_to_power,
                        project_box). USE THESE — do not re-derive
                        SINR/MI inline.
  soop1_comm.py         Algorithm 1.
  soop2_sense.py        Algorithm 2.
  moop.py               Algorithm 3.

experiments/            Sweep harness for Sec. VI figures.
  runner.py             run_one_trial / run_sweep + JSON saving.
  exp_*.py              One file per parameter sweep.

plot_results.py         Reads JSON in results/ and renders matplotlib figs.
main.py                 CLI dispatcher: `python main.py {smoke|power|...}`.
results/                Auto-generated JSON. Do not commit.
```

---

## Manuscript ↔ code anchor table

When asked to implement / fix something tied to a specific equation,
this is the table that grounds the change. Equation numbers refer to
the IEEE manuscript.

| Manuscript                    | Code location                                   |
|------------------------------ |------------------------------------------------ |
| Eq. (1) Φ_{m,i}               | `system_model/rhs.py::_phase_response`          |
| Eq. (2)–(3) Φ matrix          | `system_model/rhs.py::build_phase_matrix`       |
| Eq. (5) received signal       | implicit in `algorithms/utils.py::compute_sinr` |
| Eq. (7)–(8) sum-rate, SINR_k  | `algorithms/utils.py::sum_rate / compute_sinr`  |
| Eq. (11) CU channel           | `system_model/channel.py::generate_channel_vector` |
| Eq. (12) target response G[s] | `system_model/channel.py::generate_target_response` |
| Eq. (13)–(19) near-field b()  | `system_model/channel.py::near_field_array_response` |
| Eq. (35) sensing MI (asymptotic) | `algorithms/utils.py::sensing_mi`            |
| Eq. (51), (54) µ\*, ξ\*       | inside `_sp2_inner` (SOOP1) and `_solve_SP6` (MOOP) |
| Eq. (61) SDP (SP3')           | `algorithms/soop2_sense.py::_sp3_sdp`           |
| Eq. (65) ZF + water-filling   | `algorithms/soop1_comm.py::_zf_waterfilling`    |
| Eq. (71)–(72) Q̄, ū           | inside `_sp2_inner` and `_solve_SP6`            |
| Eq. (78)–(80) Nesterov + PGD  | inside `_sp2_inner`                             |
| Eq. (81) EVD W = U Λ^(1/2)    | `algorithms/soop2_sense.py::_omega_to_W`        |
| Eq. (87)–(88) sensing PGD     | `algorithms/soop2_sense.py::_sp4_inner`         |
| Eq. (91)–(92) SP5 (CVX)       | `algorithms/moop.py::_solve_SP5`                |
| Eq. (94)–(105) SP6 (dual)     | `algorithms/moop.py::_solve_SP6`                |
| Appendix A (Taylor expansion) | implicit in `near_field_array_response`         |

---

## Conventions you MUST follow

1. **Decoupling rule.** `system_model/` never imports from `algorithms/`.
   Algorithms never re-derive geometry/channels — they only consume a
   `Scenario` object. Any new shared quantity goes in `Scenario`.

2. **Power constraint is on the radiated signal**: `Tr(F W W^H F^H) ≤ Pt`
   (manuscript Eq. 44b). NOT on `Tr(W W^H)`. There is a known
   class of bugs where these get conflated; `utils.scale_W_to_power(F, W, Pt)`
   is the only correct way to enforce it. `_zf_waterfilling` accepts
   `(H, F, sigma2, Pt)` precisely so it can compute the per-stream
   radiated power `c_k = ||F W̄_k||^2`.

3. **Units**: `sum_rate` and `sensing_mi` return **bits/Hz** (log₂).
   `R_star`, `I_star`, and Tchebycheff τ all use the same unit. Inside
   CVX expressions for SP5, log-sum is in nats — multiply `R_star` and
   `I_star` by `ln(2)` when comparing. Easy place to introduce a unit
   bug.

4. **Auxiliary variables µ\*, ξ\* in FP-based surrogates are updated
   ONCE per outer iteration, not per inner step.** Per-step updates
   silently change the surrogate function and break monotonic ascent.
   See the long comment at the top of `_sp2_inner` for context.

5. **Outer-loop monotone safeguard.** All three solvers keep a
   "best iterate" and only return that. SOOP1 tracks best `R`, SOOP2
   tracks best `I`, MOOP tracks best τ. Do not remove this — without
   it, results have shown non-monotonic histories (see git log:
   "fix bugs causing non-monotone sum_rate").

6. **Nesterov momentum must be projected.** When applying
   `v = a + ι(a − a_prev)` before any PGD step, project `v` to `[0,1]`.
   Un-projected momentum can leave the feasible set and inflate
   `Q̄ v`, causing oscillation.

7. **Lipschitz constant**: use `_power_iter_lambda_max` for `λ_max(Q̄)`,
   NOT `||Q̄||_F`. Frobenius is a (loose) upper bound that makes the
   step size 1/λ too small and stalls convergence.

8. **CVXPY may be missing.** `algorithms/soop2_sense.py` and
   `algorithms/moop.py` import it inside a try/except and provide
   PGD-based fallbacks. Keep this dual path. Never raise on missing
   CVX — fall back gracefully.

9. **Result format is stable.** Every solver returns
   `{"W", "A", "a", "F", "history": {"sum_rate": [...], "sensing_mi": [...], ...}}`.
   The runner and plotter rely on this schema. Add new fields, never
   rename existing ones.

10. **JSON results are the artefact of record.** When changing an
    algorithm in a way that affects numbers, rerun affected experiments
    and overwrite the JSONs in `results/`. Do not edit JSONs by hand.

---

## Things NOT to do

- Don't add the manuscript PDF or any direct quotes from it to the repo.
- Don't introduce a new ML / DL dependency (torch, jax) — this is a
  pure numpy + cvxpy + scipy project on purpose.
- Don't replace the dataclass `SystemConfig` with kwargs scattered
  across functions. Every parameter the user might sweep lives in
  `config.py`.
- Don't use `localStorage` / browser-side anything; this is a
  command-line numpy project.
- Don't print large arrays. The user is debugging algorithms — print
  scalar metrics, lengths, and shapes only.

---

## How to run

```bash
# install
pip install numpy scipy cvxpy matplotlib

# fast end-to-end smoke test (8x8 array, 4 outer iters)
python main.py smoke

# the five Section VI experiments
python main.py power            # R, I vs Pt
python main.py arraysize        # R, I vs Mt
python main.py userscale        # R, I vs Kc and Ks
python main.py pareto           # Pareto frontier
python main.py convergence      # convergence curves + RHS structure compare

# plot
python plot_results.py results/exp_pareto_<timestamp>.json
```

A clean smoke test on this codebase produces SOOP1 and SOOP2 with
**zero monotonicity violations** in the printed history, and MOOP τ
strictly increasing. If you change an algorithm and a smoke run breaks
that property, treat it as a regression.

---

## When the user says "the manuscript says ..."

Treat the user's quoted snippet as ground truth for that turn but do
not write it into a file. If a fix needs a longer quote than the user
gave you, ask them to paste the relevant equation/paragraph rather
than guessing.

---

## TODO (current open items, edit me as work progresses)

- [ ] Implement Sec. V-D complexity & convergence analysis as a
      benchmarking script (CPU time + iter count vs Mt, N).
- [ ] Add baseline algorithms (random-A, all-on-A, far-field DFT
      codebook) for the comparison figures in Sec. VI.
- [ ] Verify the asymptotic-orthogonality assumption (Prop. 1)
      empirically by plotting `|b_i^H b_j|` vs Mr.
- [ ] Add a clutter-on/clutter-off ablation to confirm Eq. (20)
      reduces correctly when Ke = 0.
