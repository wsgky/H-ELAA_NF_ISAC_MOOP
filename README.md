# Holographic-ELAA Near-Field ISAC Simulation

Python implementation of the system model and beamforming algorithms
described in your manuscript.

## Project layout

```
isac_sim/
├── config.py              Global parameters (SystemConfig, AlgorithmConfig)
├── system_model/          PART 1 — scenario generation
│   ├── geometry.py        Element / feed positions
│   ├── rhs.py             Phi (Eq. 1-3), F = A Phi
│   ├── channel.py         Near-field array response (Eq. 13-19),
│   │                      CU channel (Eq. 11), target response (Eq. 12)
│   └── scenario.py        Scenario container
├── algorithms/            PART 2 — optimisation
│   ├── utils.py           Sum-rate (Eq. 7-8), sensing MI (Eq. 35), etc.
│   ├── soop1_comm.py      Algorithm 1  (ZF + water-filling +
│   │                      accelerated non-homogeneous quadratic transform)
│   ├── soop2_sense.py     Algorithm 2  (CVX SDP + EVD + PGD)
│   └── moop.py            Algorithm 3  (Tchebycheff + FP + Lagrangian dual)
├── experiments/           Sweep scripts for Section VI
│   ├── runner.py          Trial loop, JSON saving
│   ├── exp_power.py       Performance vs Pt (dBm)
│   ├── exp_arraysize.py   Performance vs Mt
│   ├── exp_userscale.py   Performance vs Kc / Ks
│   ├── exp_pareto.py      Pareto frontier (omega1, omega2)
│   └── exp_convergence.py Convergence + sub-array vs fully-connected
├── results/               Auto-saved JSON
└── main.py                CLI entry
```

## Decoupling

The simulation is split so that the **system model** (Part 1) can be
debugged completely independently of the **optimisation algorithms**
(Part 2):

- `system_model/scenario.py` returns a `Scenario` dataclass containing
  every quantity the algorithms need (`H`, `Phi`, `Bt_s`, `Br_s`,
  `gamma_s2`, …). Once a scenario is built, you can pickle it and
  hand it to any algorithm.
- The algorithms only consume the `Scenario` object — they never
  re-derive geometry or channels themselves.
- `algorithms/utils.py` exposes pure functions (`sum_rate`,
  `sensing_mi`, `compute_F`, `transmit_power`, …) so you can sanity-
  check any (W, A) you produce without going through the full pipeline.

## Dependencies

- numpy
- scipy
- cvxpy (for SP3' SDP and SP5 of MOOP — strongly recommended)

If `cvxpy` is missing, `solve_SOOP2` and `solve_MOOP` automatically
fall back to a projected-gradient heuristic so the pipeline still
runs end-to-end (results are then approximate).

```
pip install numpy scipy cvxpy
```

## Running

Quick sanity check:
```
python main.py smoke
```

Reproduce the figures in Section VI:
```
python main.py power          # Fig: R, I vs Pt
python main.py arraysize      # Fig: R, I vs Mt
python main.py userscale      # Fig: R, I vs Kc / Ks
python main.py pareto         # Fig: Pareto frontier
python main.py convergence    # Fig: convergence curves +
                              #      sub-array vs fully-connected
python main.py all
```

Each experiment writes a self-contained JSON file under `results/`
that includes the full sweep configuration plus mean/std across trials,
so plots can be regenerated later without re-running the simulation.

## Mapping back to the manuscript

| Code symbol                       | Manuscript |
|-----------------------------------|------------|
| `Phi` in `rhs.py`                 | Eq. (1)–(3)|
| `F = a[:, None] * Phi`            | F = A Phi  |
| `near_field_array_response`       | Eq. (13)–(19) |
| `generate_channel_vector`         | Eq. (11)   |
| `generate_target_response`        | Eq. (12)   |
| `compute_sinr` / `sum_rate`       | Eq. (7)–(8)|
| `sensing_mi`                      | Proposition 1, Eq. (35) |
| `_zf_waterfilling` (SOOP1)        | Eq. (63)–(65) |
| `_sp2_inner` (SOOP1)              | Lemma 1, Eq. (51), (54), (71)–(72), (78)–(80) |
| `_sp3_sdp` + `_omega_to_W` (SOOP2)| Eq. (61), (81) |
| `_sp4_inner` (SOOP2)              | Eq. (83)–(88) |
| `_solve_SP5` (MOOP)               | Eq. (91)–(92) |
| `_solve_SP6` (MOOP)               | Eq. (94)–(105) |

## Extensibility

- **Add a new sweep**: copy `experiments/exp_power.py`, change the
  `param_name` and values, you're done.
- **Add a baseline (e.g. random-A, all-on amplitude)**: write a new
  function in `algorithms/` that takes a `Scenario` and returns the
  same `(W, A, history)` dict — the runner picks it up via the
  `methods` argument.
- **Tweak channel model**: only `system_model/channel.py` changes;
  the algorithms remain untouched.
- **Tweak optimisation step sizes / iterations**: `AlgorithmConfig`
  in `config.py`.
