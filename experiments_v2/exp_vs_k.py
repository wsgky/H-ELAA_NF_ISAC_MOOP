"""
exp_vs_k.py -- Sec. V-E: R, I vs number of CUs (Kc) / STs (Ks).

Two independent sweeps (mirrors experiments/exp_ks_sensing.py):
  (a) Kc over cfg.Kc_grid (default [2,3,4,5,6]), Ks = cfg.Ks_fixed_for_kc
      (default 2) held fixed.
  (b) Ks over cfg.Ks_grid (default [1,2,3,4]), Kc = cfg.Kc_fixed_for_ks
      (default 4) held fixed.

omega1 = cfg.fixed_w1 fixed for both sweeps. N stays fixed at
cfg.sys_cfg.N (= 8) for every point (see config_v2 module docstring for
why N != Kc+Ks here). Array geometry/channels depend on Kc/Ks, so a
fresh scenario is generated per (seed, Kc, Ks).

Output array shapes (n_kc = len(cfg.Kc_grid), n_ks = len(cfg.Ks_grid),
n_mc = cfg.n_mc):
  x_kc, x_ks               : ndarray (n_kc,), (n_ks,)
  R_kc, I_kc               : dict[scheme -> ndarray (n_kc,)]  -- mean over n_mc
  R_ks, I_ks               : dict[scheme -> ndarray (n_ks,)]  -- mean over n_mc
  R_star_kc, I_star_kc     : ndarray (n_kc,)                  -- mean over n_mc
  R_star_ks, I_star_ks     : ndarray (n_ks,)                  -- mean over n_mc
"""
import copy
import numpy as np

from . import _base
from . import schemes as sch


def _sweep(cfg, sweep_name, x_grid, fixed_kc, fixed_ks, is_kc_sweep):
    sys_cfg, alg_cfg = cfg.sys_cfg, cfg.alg_cfg
    w1 = cfg.fixed_w1
    scheme_names = cfg.scheme_list()

    def _trial(scenario_base, scen_cfg_base, seed):
        out_R = {name: np.zeros(len(x_grid)) for name in scheme_names}
        out_I = {name: np.zeros(len(x_grid)) for name in scheme_names}
        out_Rstar = np.zeros(len(x_grid))
        out_Istar = np.zeros(len(x_grid))

        for xi, x in enumerate(x_grid):
            scen_cfg = copy.deepcopy(scen_cfg_base)
            if is_kc_sweep:
                scen_cfg.Kc, scen_cfg.Ks = int(x), fixed_ks
            else:
                scen_cfg.Kc, scen_cfg.Ks = fixed_kc, int(x)
            scenario, scen_cfg = _base.make_scenario(scen_cfg, seed)

            np.random.seed(seed)
            ref = sch.build_ref(scenario, scen_cfg, alg_cfg)
            out_Rstar[xi] = ref["R_star"]
            out_Istar[xi] = ref["I_star"]

            for name in scheme_names:
                np.random.seed(seed)
                R_val, I_val = sch.run_scheme(
                    name, scenario, scen_cfg, alg_cfg, w1, ref)
                out_R[name][xi] = R_val
                out_I[name][xi] = I_val

            print(f"  [exp_vs_k:{sweep_name}] seed={seed} "
                  f"Kc={scen_cfg.Kc} Ks={scen_cfg.Ks}: "
                  f"R*={ref['R_star']:.2f} I*={ref['I_star']:.2f}")

        return out_R, out_I, out_Rstar, out_Istar

    results = _base.monte_carlo(_trial, cfg.n_mc, cfg.base_seed, sys_cfg,
                                 smoke=cfg.smoke, label=f"vs_k[{sweep_name}]")

    n = len(results)
    R_mean = {name: np.mean([r[0][name] for r in results], axis=0)
              for name in scheme_names}
    I_mean = {name: np.mean([r[1][name] for r in results], axis=0)
              for name in scheme_names}
    R_star = np.mean([r[2] for r in results], axis=0)
    I_star = np.mean([r[3] for r in results], axis=0)
    return R_mean, I_mean, R_star, I_star, n


def run(cfg) -> dict:
    kc_grid = list(cfg.Kc_grid)
    ks_grid = list(cfg.Ks_grid)

    R_kc, I_kc, Rstar_kc, Istar_kc, n_kc = _sweep(
        cfg, "Kc", kc_grid,
        fixed_kc=None, fixed_ks=cfg.Ks_fixed_for_kc, is_kc_sweep=True)
    R_ks, I_ks, Rstar_ks, Istar_ks, n_ks = _sweep(
        cfg, "Ks", ks_grid,
        fixed_kc=cfg.Kc_fixed_for_ks, fixed_ks=None, is_kc_sweep=False)

    return {
        "exp": "vs_k",
        "schemes": cfg.scheme_list(),
        "x_kc": np.asarray(kc_grid, dtype=float),
        "x_ks": np.asarray(ks_grid, dtype=float),
        "x_label": "$K_c$ / $K_s$",
        "R_kc": R_kc, "I_kc": I_kc,
        "R_ks": R_ks, "I_ks": I_ks,
        "R_star_kc": Rstar_kc, "I_star_kc": Istar_kc,
        "R_star_ks": Rstar_ks, "I_star_ks": Istar_ks,
        "meta": {
            "sys_cfg": cfg.sys_cfg, "alg_cfg": cfg.alg_cfg,
            "n_mc": n_kc, "base_seed": cfg.base_seed,
            "fixed_w1": cfg.fixed_w1,
            "Kc_grid": kc_grid, "Ks_fixed_for_kc": cfg.Ks_fixed_for_kc,
            "Ks_grid": ks_grid, "Kc_fixed_for_ks": cfg.Kc_fixed_for_ks,
            "timestamp": _base.timestamp(), "git_note": "v2",
        },
    }
