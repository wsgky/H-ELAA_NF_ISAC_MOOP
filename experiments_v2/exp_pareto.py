"""
exp_pareto.py -- Pareto-frontier data (Sec. V-D).

For each scheme in cfg.scheme_list(), sweep omega1 over
cfg.weights_pareto (default np.linspace(0,1,21)) and record the final
(R, I) operating point of that scheme, averaged over cfg.n_mc
Monte-Carlo scenarios.

"fully_digital" does not depend on omega1 (Sec. V-B): it is evaluated
once per scenario and broadcast to every weight column.

omega1 in {0, 1} (the endpoints of np.linspace(0,1,21)) is clipped to
``_base.PARETO_W1_EPS`` before being passed to the solver, since
``_solve_SP5_v2``/``_solve_SP6`` (algorithms/moop.py) divide by
omega1/omega2 (Eq. 63b/63c, 73b/73c, 76-78) and raise ZeroDivisionError
at the exact boundary. The recorded ``x`` values remain the nominal
0/1 (only the value passed to the solver is clipped).

Output array shapes (n_w = len(cfg.weights_pareto), n_mc = cfg.n_mc):
  x              : ndarray (n_w,)              -- omega1 values
  R, I           : dict[scheme -> ndarray (n_w,)]  -- mean over n_mc
  R_star, I_star : ndarray (n_mc,)             -- SOOP1/SOOP2 references
"""
import numpy as np

from . import _base
from . import schemes as sch


def run(cfg) -> dict:
    sys_cfg, alg_cfg = cfg.sys_cfg, cfg.alg_cfg
    weights = np.asarray(cfg.weights_pareto, dtype=float)
    scheme_names = cfg.scheme_list()

    def _trial(scenario, scen_cfg, seed):
        np.random.seed(seed)
        ref = sch.build_ref(scenario, scen_cfg, alg_cfg)

        out_R = {}
        out_I = {}
        for name in scheme_names:
            if name == "fully_digital":
                # weight-independent: evaluate once, broadcast
                np.random.seed(seed)
                R_val, I_val = sch.run_scheme(
                    name, scenario, scen_cfg, alg_cfg, 0.5, ref)
                out_R[name] = np.full(len(weights), R_val)
                out_I[name] = np.full(len(weights), I_val)
                print(f"  [exp_pareto] seed={seed} scheme={name}: "
                      f"R={R_val:.3f} I={I_val:.3f}")
                continue

            R_row = np.zeros(len(weights))
            I_row = np.zeros(len(weights))
            for wi, w1 in enumerate(weights):
                np.random.seed(seed)
                # omega1 in {0,1} -> ZeroDivisionError inside SP5/SP6
                # (algorithms/moop.py); clip to the open interval.
                w1_eff = _base.safe_w1(float(w1))
                R_val, I_val = sch.run_scheme(
                    name, scenario, scen_cfg, alg_cfg, w1_eff, ref)
                R_row[wi], I_row[wi] = R_val, I_val
            out_R[name] = R_row
            out_I[name] = I_row
            print(f"  [exp_pareto] seed={seed} scheme={name}: "
                  f"R in [{R_row.min():.2f},{R_row.max():.2f}]  "
                  f"I in [{I_row.min():.2f},{I_row.max():.2f}]")

        return out_R, out_I, ref["R_star"], ref["I_star"]

    results = _base.monte_carlo(_trial, cfg.n_mc, cfg.base_seed, sys_cfg,
                                 smoke=cfg.smoke, label="pareto")

    n = len(results)
    R_mean = {name: np.mean([r[0][name] for r in results], axis=0)
              for name in scheme_names}
    I_mean = {name: np.mean([r[1][name] for r in results], axis=0)
              for name in scheme_names}
    R_star = np.asarray([r[2] for r in results])
    I_star = np.asarray([r[3] for r in results])

    return {
        "exp": "pareto",
        "schemes": scheme_names,
        "x": weights,
        "x_label": r"$\omega_1$",
        "R": R_mean,
        "I": I_mean,
        "R_star": R_star,
        "I_star": I_star,
        "meta": {
            "sys_cfg": sys_cfg, "alg_cfg": alg_cfg,
            "n_mc": n, "base_seed": cfg.base_seed,
            "weights": weights,
            "timestamp": _base.timestamp(), "git_note": "v2",
        },
    }
