"""
exp_vs_mt.py -- Sec. V-E: R, I vs TX/RX array size Mt_h (Mt_v fixed).

Sweeps Mt_h over cfg.Mt_h_grid (default [32,64,96,128]) with
Mt_v = Mr_v = cfg.Mt_v_fixed and Mr_h = Mt_h, omega1 = cfg.fixed_w1
fixed. Array geometry/channels depend on Mt, so a fresh scenario is
generated per (seed, Mt_h) -- still seeded by the Monte-Carlo index for
reproducibility.

N (= cfg.sys_cfg.N, fixed at 8) divides every Mt = Mt_h * Mt_v_fixed
since Mt_v_fixed is itself a multiple of 8 (16 normally, 8 in --smoke).

Also reports the SOOP1/SOOP2 single-objective references R*(Mt), I*(Mt).

Output array shapes (n_mt = len(cfg.Mt_h_grid), n_mc = cfg.n_mc):
  x              : ndarray (n_mt,)  -- Mt_h values
  R, I           : dict[scheme -> ndarray (n_mt,)]  -- mean over n_mc
  R_star, I_star : ndarray (n_mt,)                  -- mean over n_mc
"""
import copy
import numpy as np

from . import _base
from . import schemes as sch


def run(cfg) -> dict:
    sys_cfg, alg_cfg = cfg.sys_cfg, cfg.alg_cfg
    mt_h_grid = list(cfg.Mt_h_grid)
    mt_v = cfg.Mt_v_fixed
    w1 = cfg.fixed_w1
    scheme_names = cfg.scheme_list()

    def _trial(scenario_base, scen_cfg_base, seed):
        out_R = {name: np.zeros(len(mt_h_grid)) for name in scheme_names}
        out_I = {name: np.zeros(len(mt_h_grid)) for name in scheme_names}
        out_Rstar = np.zeros(len(mt_h_grid))
        out_Istar = np.zeros(len(mt_h_grid))

        for mi, mt_h in enumerate(mt_h_grid):
            scen_cfg = copy.deepcopy(scen_cfg_base)
            scen_cfg.Mt_h, scen_cfg.Mt_v = int(mt_h), mt_v
            scen_cfg.Mr_h, scen_cfg.Mr_v = int(mt_h), mt_v
            scenario, scen_cfg = _base.make_scenario(scen_cfg, seed)

            np.random.seed(seed)
            ref = sch.build_ref(scenario, scen_cfg, alg_cfg)
            out_Rstar[mi] = ref["R_star"]
            out_Istar[mi] = ref["I_star"]

            for name in scheme_names:
                np.random.seed(seed)
                R_val, I_val = sch.run_scheme(
                    name, scenario, scen_cfg, alg_cfg, w1, ref)
                out_R[name][mi] = R_val
                out_I[name][mi] = I_val

            print(f"  [exp_vs_mt] seed={seed} Mt_h={mt_h} (Mt={scenario.Mt}): "
                  f"R*={ref['R_star']:.2f} I*={ref['I_star']:.2f}")

        return out_R, out_I, out_Rstar, out_Istar

    results = _base.monte_carlo(_trial, cfg.n_mc, cfg.base_seed, sys_cfg,
                                 smoke=cfg.smoke, label="vs_mt")

    n = len(results)
    R_mean = {name: np.mean([r[0][name] for r in results], axis=0)
              for name in scheme_names}
    I_mean = {name: np.mean([r[1][name] for r in results], axis=0)
              for name in scheme_names}
    R_star = np.mean([r[2] for r in results], axis=0)
    I_star = np.mean([r[3] for r in results], axis=0)

    return {
        "exp": "vs_mt",
        "schemes": scheme_names,
        "x": np.asarray(mt_h_grid, dtype=float),
        "x_label": "$M_{t,h}$ ($M_{t,v}=%d$)" % mt_v,
        "R": R_mean,
        "I": I_mean,
        "R_star": R_star,
        "I_star": I_star,
        "meta": {
            "sys_cfg": sys_cfg, "alg_cfg": alg_cfg,
            "n_mc": n, "base_seed": cfg.base_seed,
            "fixed_w1": w1, "Mt_h_grid": mt_h_grid, "Mt_v_fixed": mt_v,
            "timestamp": _base.timestamp(), "git_note": "v2",
        },
    }
