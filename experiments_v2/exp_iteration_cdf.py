"""
exp_iteration_cdf.py -- Fig. 4 data: distribution (over cfg.n_mc seeded
scenarios) of the number of outer iterations needed for tau to converge,
for omega1 in cfg.weights_conv (default [0.2, 0.5, 0.8]).

Convergence criterion (manuscript epsilon, ``cfg.eps_tau``):
    |tau^(s) - tau^(s-1)| <= eps_tau,   s = 1..Smax  (capped at Smax)

The MOOP outer-loop early-stop patience is disabled
(``config_v2.full_trajectory_alg_cfg``) so the full ``Smax``-length tau
trajectory is always available to evaluate this criterion. The CDF itself
is computed in plot_v2.py from ``iter_counts``.

Output array shapes (n_w = len(cfg.weights_conv), n_mc = cfg.n_mc):
  iter_counts : ndarray (n_w, n_mc) int  -- # outer iters until convergence
  weights     : list[float], length n_w
  Smax, eps_tau : also stored in meta
"""
import numpy as np

from . import _base
from . import schemes as sch
from config_v2 import full_trajectory_alg_cfg


def run(cfg) -> dict:
    sys_cfg, alg_cfg = cfg.sys_cfg, cfg.alg_cfg
    alg_cfg_full = full_trajectory_alg_cfg(alg_cfg)
    Smax = alg_cfg_full.MOOP_outer_iters
    eps_tau = cfg.eps_tau
    weights = list(cfg.weights_conv)

    def _trial(scenario, scen_cfg, seed):
        np.random.seed(seed)
        ref = sch.build_ref(scenario, scen_cfg, alg_cfg)

        counts = np.zeros(len(weights), dtype=int)
        for wi, w1 in enumerate(weights):
            np.random.seed(seed)
            res = _base.solve_proposed(
                scenario, scen_cfg, alg_cfg_full, w1,
                R_star=ref["R_star"], I_star=ref["I_star"],
                soop1_result=ref["soop1"], soop2_result=ref["soop2"])
            tau_seq = np.asarray(res["history"]["tau"], dtype=float)  # len Smax+1
            d_tau = np.abs(np.diff(tau_seq))                           # len Smax
            below = np.where(d_tau <= eps_tau)[0]
            counts[wi] = int(below[0] + 1) if len(below) else Smax
            print(f"  [exp_iteration_cdf] seed={seed} w1={w1}: "
                  f"iters={counts[wi]}")
        return counts

    results = _base.monte_carlo(_trial, cfg.n_mc, cfg.base_seed, sys_cfg,
                                 smoke=cfg.smoke, label="iteration_cdf")

    iter_counts = np.stack(results, axis=1)  # (n_w, n_mc)

    return {
        "exp": "iteration_cdf",
        "schemes": ["proposed"],
        "x": None,
        "x_label": "Number of outer iterations to converge",
        "weights": weights,
        "iter_counts": iter_counts,
        "meta": {
            "sys_cfg": sys_cfg, "alg_cfg": alg_cfg_full,
            "n_mc": iter_counts.shape[1], "base_seed": cfg.base_seed,
            "weights": weights, "Smax": Smax, "eps_tau": eps_tau,
            "timestamp": _base.timestamp(), "git_note": "v2",
        },
    }
