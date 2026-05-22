"""
Experiment: sensing MI vs transmit power Pt_dBm.

Sweeps Pt_dBm in [5, 10, 15, 20, 25, 30] with 10 Monte Carlo trials each.
Array: 128x8, N=8, Kc=2, Ks=2, Ke=2.
Both fully_connected and subarray structures.

Baselines
---------
  noholo : a=1 (fixed), W optimised via SP3' SDP  (no holographic amplitude opt.)
  allon  : a=1, isotropic W scaled to Pt
  rand   : a ~ U[0,1], isotropic W scaled to Pt

Usage:
    python main.py power_sensing
    python experiments/exp_power_sensing.py
"""
import copy
import json
import sys
import time

import numpy as np

from config import AlgorithmConfig, SystemConfig
from experiments.runner import RESULTS_DIR, _to_jsonable
from system_model import generate_scenario
from algorithms import solve_SOOP2
from algorithms.soop2_sense import _sp3_sdp, _omega_to_W
from algorithms.utils import compute_F, sensing_mi, scale_W_to_power


# ── shared helpers ─────────────────────────────────────────────────────────────

def _W_iso(F, N, dim, Pt):
    W = np.eye(N, dim, dtype=complex)
    FW = F @ W
    p = float(np.real(np.trace(FW @ FW.conj().T)))
    if p > 1e-12:
        W *= np.sqrt(Pt / p)
    return W


def _mi_noholo(scen, cfg, solver):
    """Sensing MI: a=1, W optimised by SDP (no RHS amplitude optimisation)."""
    a = np.ones(scen.Mt)
    F = compute_F(a, scen.Phi)
    Om = _sp3_sdp(F, scen.Bt_s, scen.gamma_s2, cfg.sigma_s2,
                  cfg.L, cfg.Pt, scen.N, solver=solver)
    W = scale_W_to_power(F, _omega_to_W(Om, scen.Kc + scen.Ks), cfg.Pt)
    return float(sensing_mi(scen.Bt_s, scen.gamma_s2, F, W, cfg.sigma_s2, cfg.L, scen.Mr))


def _mi_allon(scen, cfg):
    """Sensing MI: a=1, isotropic W."""
    F = compute_F(np.ones(scen.Mt), scen.Phi)
    W = _W_iso(F, scen.N, scen.Kc + scen.Ks, cfg.Pt)
    return float(sensing_mi(scen.Bt_s, scen.gamma_s2, F, W, cfg.sigma_s2, cfg.L, scen.Mr))


def _mi_rand(scen, cfg, seed):
    """Sensing MI: a ~ U[0,1], isotropic W."""
    a = np.random.default_rng(seed).uniform(0, 1, scen.Mt)
    F = compute_F(a, scen.Phi)
    W = _W_iso(F, scen.N, scen.Kc + scen.Ks, cfg.Pt)
    return float(sensing_mi(scen.Bt_s, scen.gamma_s2, F, W, cfg.sigma_s2, cfg.L, scen.Mr))


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    Pt_list  = [5, 10, 15, 20, 25, 30]
    structs  = ["fully_connected", "subarray"]
    n_trials = 10
    SEED     = 2025

    base_cfg = SystemConfig(
        Mt_h=128, Mt_v=8, Mr_h=128, Mr_v=8, N=8,
        Kc=2, Ks=2, Ke=2, L=128, seed=SEED,
    )
    alg_cfg = AlgorithmConfig(outer_iters=10, inner_iters=100, pgd_step_a=5e-4)

    out = {"label": "exp_power_sensing",
           "Pt_dBm": Pt_list, "n_trials": n_trials,
           "fully_connected": {m: {"mean": [], "std": []}
                                for m in ["soop2", "noholo", "allon", "rand"]},
           "subarray":        {m: {"mean": [], "std": []}
                                for m in ["soop2", "noholo", "allon", "rand"]}}

    print(f"[exp_power_sensing] Mt=128x8  N=8  Kc=2  Ks=2  Ke=2  "
          f"{n_trials} trials  outer={alg_cfg.outer_iters}")
    print("-" * 65)

    t0 = time.time()
    for struct in structs:
        print(f"\n  struct = {struct}")
        for pt in Pt_list:
            vals = {m: [] for m in ["soop2", "noholo", "allon", "rand"]}
            for trial in range(n_trials):
                seed = SEED + trial + 1
                cfg  = copy.deepcopy(base_cfg)
                cfg.Pt_dBm        = pt
                cfg.rhs_structure = struct
                cfg.seed          = seed
                scen = generate_scenario(cfg, np.random.default_rng(seed))

                res = solve_SOOP2(scen, cfg, alg_cfg)
                vals["soop2"].append(max(res["history"]["sensing_mi"]))
                vals["noholo"].append(_mi_noholo(scen, cfg, alg_cfg.cvx_solver))
                vals["allon"].append(_mi_allon(scen, cfg))
                vals["rand"].append(_mi_rand(scen, cfg, seed + 500))

            for m, v in vals.items():
                out[struct][m]["mean"].append(float(np.mean(v)))
                out[struct][m]["std"].append(float(np.std(v)))
            print(f"    Pt={pt:2d} dBm | "
                  + "  ".join(f"{m}={np.mean(vals[m]):.3f}" for m in vals))

    out["elapsed_sec"] = time.time() - t0
    ts  = time.strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"exp_power_sensing_{ts}.json"
    with open(path, "w") as f:
        json.dump(_to_jsonable(out), f, indent=2)
    print(f"\n[exp_power_sensing] saved -> {path}")


if __name__ == "__main__":
    main()
