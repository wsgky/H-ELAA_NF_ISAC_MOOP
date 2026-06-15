"""
exp_vs_power.py -- Sec. V-E: R, I vs transmit power Pt.

Sweeps Pt over cfg.Pt_grid_dBm (default [10,15,20,25,30,35,40] dBm) with
omega1 = cfg.fixed_w1 (default 0.5) fixed. The scenario geometry/channels
do not depend on Pt, so a single scenario per Monte-Carlo seed is reused
across the whole Pt sweep (only ``SystemConfig.Pt_dBm`` -- and hence
``sys_cfg.Pt`` -- changes).

Also reports the SOOP1/SOOP2 single-objective references R*(Pt), I*(Pt).

Output array shapes (n_pt = len(cfg.Pt_grid_dBm), n_mc = cfg.n_mc):
  x              : ndarray (n_pt,)  -- Pt in dBm
  R, I           : dict[scheme -> ndarray (n_pt,)]  -- mean over n_mc
  R_star, I_star : ndarray (n_pt,)                  -- mean over n_mc
"""
import copy
import numpy as np

from . import _base
from . import schemes as sch


def run(cfg) -> dict:
    sys_cfg, alg_cfg = cfg.sys_cfg, cfg.alg_cfg
    pt_grid = list(cfg.Pt_grid_dBm)
    w1 = cfg.fixed_w1
    scheme_names = cfg.scheme_list()

    def _trial(scenario, scen_cfg_base, seed):
        out_R = {name: np.zeros(len(pt_grid)) for name in scheme_names}
        out_I = {name: np.zeros(len(pt_grid)) for name in scheme_names}
        out_Rstar = np.zeros(len(pt_grid))
        out_Istar = np.zeros(len(pt_grid))

        for pi, pt_dbm in enumerate(pt_grid):
            scen_cfg = copy.deepcopy(scen_cfg_base)
            scen_cfg.Pt_dBm = float(pt_dbm)

            np.random.seed(seed)
            ref = sch.build_ref(scenario, scen_cfg, alg_cfg)
            out_Rstar[pi] = ref["R_star"]
            out_Istar[pi] = ref["I_star"]

            for name in scheme_names:
                np.random.seed(seed)
                R_val, I_val = sch.run_scheme(
                    name, scenario, scen_cfg, alg_cfg, w1, ref)
                out_R[name][pi] = R_val
                out_I[name][pi] = I_val

            print(f"  [exp_vs_power] seed={seed} Pt={pt_dbm}dBm: "
                  f"R*={ref['R_star']:.2f} I*={ref['I_star']:.2f}")

        return out_R, out_I, out_Rstar, out_Istar

    results = _base.monte_carlo(_trial, cfg.n_mc, cfg.base_seed, sys_cfg,
                                 smoke=cfg.smoke, label="vs_power")

    n = len(results)
    R_mean = {name: np.mean([r[0][name] for r in results], axis=0)
              for name in scheme_names}
    I_mean = {name: np.mean([r[1][name] for r in results], axis=0)
              for name in scheme_names}
    R_star = np.mean([r[2] for r in results], axis=0)
    I_star = np.mean([r[3] for r in results], axis=0)

    return {
        "exp": "vs_power",
        "schemes": scheme_names,
        "x": np.asarray(pt_grid, dtype=float),
        "x_label": "$P_t$ [dBm]",
        "R": R_mean,
        "I": I_mean,
        "R_star": R_star,
        "I_star": I_star,
        "meta": {
            "sys_cfg": sys_cfg, "alg_cfg": alg_cfg,
            "n_mc": n, "base_seed": cfg.base_seed,
            "fixed_w1": w1, "Pt_grid_dBm": pt_grid,
            "timestamp": _base.timestamp(), "git_note": "v2",
        },
    }
