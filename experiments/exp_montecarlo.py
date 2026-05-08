"""
Monte Carlo validation experiment.

Fixed system parameters, N_mc independent random channel realisations.
Reports mean ± std of sum_rate and sensing_mi for SOOP1, SOOP2, and
MOOP (three representative omega pairs).

Usage:
    python main.py montecarlo           # default 50 trials
    python main.py montecarlo 100       # 100 trials
    python experiments/exp_montecarlo.py 50
"""
import json
import sys
import time

import numpy as np

from config import AlgorithmConfig, SystemConfig
from experiments.runner import RESULTS_DIR, _to_jsonable, run_one_trial


def main(n_trials: int = 50):
    sys_cfg = SystemConfig(
        Mt_h=16, Mt_v=16, Mr_h=16, Mr_v=16, N=8,
        Kc=3, Ks=2, Ke=2, L=128, seed=2025,
        rhs_structure="subarray",
    )
    alg_cfg = AlgorithmConfig(outer_iters=8, inner_iters=15)
    omega_list = [(0.2, 0.8), (0.5, 0.5), (0.8, 0.2)]

    print(f"[exp_montecarlo] {n_trials} trials | "
          f"Mt={sys_cfg.Mt}, Mr={sys_cfg.Mr}, "
          f"Kc={sys_cfg.Kc}, Ks={sys_cfg.Ks}, Ke={sys_cfg.Ke}, "
          f"Pt={sys_cfg.Pt_dBm} dBm")
    print(f"  alg: outer={alg_cfg.outer_iters}, inner={alg_cfg.inner_iters}")
    print("-" * 72)

    raw = []
    t0 = time.time()

    for trial in range(n_trials):
        seed = sys_cfg.seed + trial + 1
        res = run_one_trial(sys_cfg, alg_cfg, seed,
                            methods=("SOOP1", "SOOP2", "MOOP"),
                            omega_list=omega_list)
        raw.append(res)

        r1 = res["SOOP1"]["sum_rate"]
        i1 = res["SOOP1"]["sensing_mi"]
        r2 = res["SOOP2"]["sum_rate"]
        i2 = res["SOOP2"]["sensing_mi"]
        rm = res["MOOP"][1]["sum_rate"]   # omega=(0.5, 0.5)
        im = res["MOOP"][1]["sensing_mi"]
        print(f"  trial {trial + 1:3d}/{n_trials} | "
              f"SOOP1 R={r1:.3f} I={i1:.3f} | "
              f"SOOP2 R={r2:.3f} I={i2:.3f} | "
              f"MOOP(0.5/0.5) R={rm:.3f} I={im:.3f}")

    elapsed = time.time() - t0

    # ── aggregate ──────────────────────────────────────────────────────────
    R1 = np.array([r["SOOP1"]["sum_rate"]   for r in raw])
    I1 = np.array([r["SOOP1"]["sensing_mi"] for r in raw])
    R2 = np.array([r["SOOP2"]["sum_rate"]   for r in raw])
    I2 = np.array([r["SOOP2"]["sensing_mi"] for r in raw])

    soop1_stats = {
        "sum_rate_mean":    float(R1.mean()),
        "sum_rate_std":     float(R1.std()),
        "sensing_mi_mean":  float(I1.mean()),
        "sensing_mi_std":   float(I1.std()),
        "sum_rate_all":     R1.tolist(),
        "sensing_mi_all":   I1.tolist(),
    }
    soop2_stats = {
        "sum_rate_mean":    float(R2.mean()),
        "sum_rate_std":     float(R2.std()),
        "sensing_mi_mean":  float(I2.mean()),
        "sensing_mi_std":   float(I2.std()),
        "sum_rate_all":     R2.tolist(),
        "sensing_mi_all":   I2.tolist(),
    }
    moop_stats = []
    for o, (w1, w2) in enumerate(omega_list):
        Rm = np.array([r["MOOP"][o]["sum_rate"]   for r in raw])
        Im = np.array([r["MOOP"][o]["sensing_mi"] for r in raw])
        moop_stats.append({
            "omega1": w1, "omega2": w2,
            "sum_rate_mean":    float(Rm.mean()),
            "sum_rate_std":     float(Rm.std()),
            "sensing_mi_mean":  float(Im.mean()),
            "sensing_mi_std":   float(Im.std()),
            "sum_rate_all":     Rm.tolist(),
            "sensing_mi_all":   Im.tolist(),
        })

    # ── print summary table ────────────────────────────────────────────────
    W = 72
    print("\n" + "=" * W)
    print(f"  Monte Carlo Summary  ({n_trials} trials, {elapsed:.1f} s)")
    print("=" * W)
    print(f"  {'Algorithm':<22}  {'R mean':>8}  {'R std':>7}  "
          f"{'I mean':>8}  {'I std':>7}")
    print("  " + "-" * (W - 2))

    def row(name, st):
        print(f"  {name:<22}  {st['sum_rate_mean']:>8.4f}  "
              f"{st['sum_rate_std']:>7.4f}  "
              f"{st['sensing_mi_mean']:>8.4f}  "
              f"{st['sensing_mi_std']:>7.4f}")

    row("SOOP1", soop1_stats)
    row("SOOP2", soop2_stats)
    for ms in moop_stats:
        row(f"MOOP(w1={ms['omega1']:.1f},w2={ms['omega2']:.1f})", ms)

    print("=" * W)
    print(f"  Units: bits/Hz  |  R = sum-rate, I = sensing MI")
    print("=" * W + "\n")

    # ── save JSON ──────────────────────────────────────────────────────────
    pack = {
        "label": "exp_montecarlo",
        "n_trials": n_trials,
        "sys_cfg": {
            "Mt_h": sys_cfg.Mt_h, "Mt_v": sys_cfg.Mt_v,
            "Mr_h": sys_cfg.Mr_h, "Mr_v": sys_cfg.Mr_v,
            "N": sys_cfg.N, "Kc": sys_cfg.Kc,
            "Ks": sys_cfg.Ks, "Ke": sys_cfg.Ke,
            "L": sys_cfg.L, "Pt_dBm": sys_cfg.Pt_dBm,
            "rhs_structure": sys_cfg.rhs_structure,
        },
        "alg_cfg": {
            "outer_iters": alg_cfg.outer_iters,
            "inner_iters": alg_cfg.inner_iters,
        },
        "omega_list": omega_list,
        "summary": {
            "SOOP1": soop1_stats,
            "SOOP2": soop2_stats,
            "MOOP": moop_stats,
        },
        "elapsed_sec": elapsed,
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"exp_montecarlo_{ts}.json"
    with open(path, "w") as f:
        json.dump(_to_jsonable(pack), f, indent=2)
    print(f"[exp_montecarlo] saved -> {path}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    main(n)
