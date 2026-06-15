"""
exp_convergence.py -- Fig. 3 data: per-outer-iteration convergence of
tau, R, I (proposed scheme / Algorithm 1) for omega1 in cfg.weights_conv
(default [0.2, 0.5, 0.8]).

The MOOP outer-loop early-stop patience is disabled
(``config_v2.full_trajectory_alg_cfg``) so every run produces the full
``Smax``-length trajectory; trajectories that would have converged earlier
are flat-extended (last value repeated) to length ``Smax``.

By default a single fixed scenario (seed = cfg.base_seed) is used; in
``--smoke`` mode ``cfg.n_mc`` seeds are averaged instead (cheap check that
averaging works).

Output array shapes (n_w = len(cfg.weights_conv), Smax = alg_cfg.MOOP_outer_iters
after config_v2.full_trajectory_alg_cfg):
  x       : ndarray (Smax,)        -- outer-iteration index s = 1..Smax
  tau     : ndarray (n_w, Smax)    -- mean over seeds
  R       : ndarray (n_w, Smax)    -- mean over seeds, bits/Hz
  I       : ndarray (n_w, Smax)    -- mean over seeds, bits/Hz
  weights : list[float], length n_w
"""
import numpy as np

from . import _base
from . import schemes as sch
from config_v2 import full_trajectory_alg_cfg


def run(cfg) -> dict:
    sys_cfg, alg_cfg = cfg.sys_cfg, cfg.alg_cfg
    alg_cfg_full = full_trajectory_alg_cfg(alg_cfg)
    Smax = alg_cfg_full.MOOP_outer_iters
    weights = list(cfg.weights_conv)

    n_seeds = cfg.n_mc if cfg.smoke else 1

    def _trial(scenario, scen_cfg, seed):
        tau_row = np.zeros((len(weights), Smax))
        R_row = np.zeros((len(weights), Smax))
        I_row = np.zeros((len(weights), Smax))

        np.random.seed(seed)
        print(f"  [exp_convergence] seed={seed}: building reference points")
        ref = sch.build_ref(scenario, scen_cfg, alg_cfg)
        print(f"  [exp_convergence] seed={seed}: building reference points finished | R_star={ref['R_star']:.3f}, I_star={ref['I_star']:.3f}")
        for wi, w1 in enumerate(weights):
            np.random.seed(seed)
            res = _base.solve_proposed(
                scenario, scen_cfg, alg_cfg_full, w1,
                R_star=ref["R_star"], I_star=ref["I_star"],
                soop1_result=ref["soop1"], soop2_result=ref["soop2"])
            hist = res["history"]
            t_seq = np.asarray(hist["tau"][1:], dtype=float)
            r_seq = np.asarray(hist["sum_rate"][1:], dtype=float)
            i_seq = np.asarray(hist["sensing_mi"][1:], dtype=float)
            n = len(t_seq)
            if n < Smax:
                t_seq = np.concatenate([t_seq, np.full(Smax - n, t_seq[-1])])
                r_seq = np.concatenate([r_seq, np.full(Smax - n, r_seq[-1])])
                i_seq = np.concatenate([i_seq, np.full(Smax - n, i_seq[-1])])
            tau_row[wi] = t_seq[:Smax]
            R_row[wi] = r_seq[:Smax]
            I_row[wi] = i_seq[:Smax]
            print(f"  [exp_convergence] seed={seed} w1={w1}: "
                  f"tau_final={t_seq[-1]:.4f} R_final={r_seq[-1]:.3f} "
                  f"I_final={i_seq[-1]:.3f}")

        return tau_row, R_row, I_row

    results = _base.monte_carlo(_trial, n_seeds, cfg.base_seed, sys_cfg,
                                 smoke=cfg.smoke, label="convergence")

    tau = np.mean([r[0] for r in results], axis=0)
    R = np.mean([r[1] for r in results], axis=0)
    I = np.mean([r[2] for r in results], axis=0)

    return {
        "exp": "convergence",
        "schemes": ["proposed"],
        "x": np.arange(1, Smax + 1),
        "x_label": "Outer iteration $s$",
        "weights": weights,
        "tau": tau,
        "R": R,
        "I": I,
        "meta": {
            "sys_cfg": sys_cfg, "alg_cfg": alg_cfg_full,
            "n_mc": len(results), "base_seed": cfg.base_seed,
            "weights": weights, "Smax": Smax,
            "timestamp": _base.timestamp(), "git_note": "v2",
        },
    }
