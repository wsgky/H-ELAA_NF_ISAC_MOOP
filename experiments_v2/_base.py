"""
Shared infrastructure for experiments_v2.

Everything here is a thin wrapper over the existing, read-only packages
``system_model`` and ``algorithms`` (see CLAUDE.md decoupling rule --
this module is allowed to depend on both, individual experiments should
only depend on this module and ``schemes``).

Functions
---------
make_scenario(sys_cfg, seed)
    Build one Monte-Carlo scenario (thin wrapper over
    ``system_model.generate_scenario``), returning ``(scenario, cfg)``
    where ``cfg`` is a deep copy of ``sys_cfg`` with ``seed`` set.

monte_carlo(fn, n_mc, base_seed, sys_cfg, smoke=False, label="")
    Runs ``fn(scenario, cfg, seed)`` over ``n_mc`` seeded scenario
    realisations and returns the list of per-trial results (one entry
    per MC trial; experiments stack these into ndarrays themselves,
    since the stacking shape is experiment-specific).

reference_points(scenario, sys_cfg, alg_cfg)
    Runs SOOP1 / SOOP2 once and returns ``(R_star, I_star, soop1_result,
    soop2_result)``.

solve_proposed(scenario, sys_cfg, alg_cfg, w1, ...)
    Runs Algorithm 1 (MOOP) for weight ``omega1=w1, omega2=1-w1``.

eval_metrics(scenario, sys_cfg, F, W)
    Returns ``(R, I)`` using the existing ``sum_rate`` / ``sensing_mi``.

save_result(name, result_dict) / load_result(name)
    pickle (+ metadata-only json) IO into ``result_v2/``.
"""
import copy
import json
import pickle
import time
from pathlib import Path

import numpy as np

from system_model import generate_scenario
from algorithms import solve_SOOP1, solve_SOOP2, solve_MOOP
from algorithms.utils import sum_rate, sensing_mi


RESULT_DIR = Path(__file__).resolve().parent.parent / "result_v2"
RESULT_DIR.mkdir(exist_ok=True)

# ``_solve_SP6`` / ``_solve_SP5_v2`` (algorithms/moop.py) divide by
# ``omega1`` and ``omega2`` in several places (Eq. 63b/63c, 73b/73c,
# 76-78 -- e.g. ``tau / omega1``, ``lam1 * R_star / omega1``), so
# omega1 in {0, 1} raises ZeroDivisionError / produces inf constraints.
# Sweeps that include the Tchebycheff weight boundary (e.g. exp_pareto's
# omega1 in {0, 1}) must clip to this epsilon before calling
# solve_proposed / SP5 / SP6.
PARETO_W1_EPS = 1e-3


def safe_w1(w1: float, eps: float = PARETO_W1_EPS) -> float:
    """Clip omega1 away from the {0, 1} boundary (see PARETO_W1_EPS)."""
    return float(np.clip(w1, eps, 1.0 - eps))


# ----------------------------------------------------------------------
# Scenario construction
# ----------------------------------------------------------------------
def make_scenario(sys_cfg, seed: int):
    """Build one seeded Monte-Carlo scenario.

    Returns
    -------
    scenario : system_model.Scenario
    cfg      : SystemConfig
        Deep copy of ``sys_cfg`` with ``cfg.seed = seed``.
    """
    cfg = copy.deepcopy(sys_cfg)
    cfg.seed = seed
    rng = np.random.default_rng(seed)
    scenario = generate_scenario(cfg, rng=rng)
    return scenario, cfg


# ----------------------------------------------------------------------
# Monte-Carlo loop
# ----------------------------------------------------------------------
def monte_carlo(fn, n_mc: int, base_seed: int, sys_cfg, smoke: bool = False,
                label: str = ""):
    """Run ``fn(scenario, cfg, seed)`` over ``n_mc`` seeded realisations.

    Parameters
    ----------
    fn : callable(scenario, cfg, seed) -> Any
    n_mc : int
        Number of Monte-Carlo trials (reduced to ``min(n_mc, 3)`` if
        ``smoke=True``).
    base_seed : int
        Trial ``i`` uses ``seed = base_seed + i``.

    Returns
    -------
    list
        One entry per MC trial, in order.
    """
    if smoke:
        n_mc = min(n_mc, 3)

    results = []
    for i in range(n_mc):
        print(f"[monte_carlo] Starting trial {i + 1}/{n_mc} (seed={base_seed + i})")
        seed = base_seed + i
        scenario, cfg = make_scenario(sys_cfg, seed)
        t0 = time.time()
        res = fn(scenario, cfg, seed)
        dt = time.time() - t0
        tag = f"[{label}] " if label else ""
        print(f"  {tag}MC {i + 1}/{n_mc} (seed={seed}) done in {dt:.1f}s")
        results.append(res)
    return results


# ----------------------------------------------------------------------
# Reference points R*, I*
# ----------------------------------------------------------------------
def reference_points(scenario, sys_cfg, alg_cfg):
    """Run SOOP1 / SOOP2 once; return (R_star, I_star, soop1_result, soop2_result)."""
    soop1_result = solve_SOOP1(scenario, sys_cfg, alg_cfg)
    soop2_result = solve_SOOP2(scenario, sys_cfg, alg_cfg)
    R_star = soop1_result["history"]["sum_rate"][-1]
    I_star = soop2_result["history"]["sensing_mi"][-1]
    return R_star, I_star, soop1_result, soop2_result


# ----------------------------------------------------------------------
# Proposed (Algorithm 1 / MOOP)
# ----------------------------------------------------------------------
def solve_proposed(scenario, sys_cfg, alg_cfg, w1: float,
                   R_star=None, I_star=None,
                   soop1_result=None, soop2_result=None,
                   full_history: bool = False):
    """Run Algorithm 1 (MOOP) for omega1=w1, omega2=1-w1.

    Returns the raw dict from ``solve_MOOP`` (W, A, a, F, history, ...).
    """
    w2 = 1.0 - w1
    return solve_MOOP(scenario, sys_cfg, alg_cfg,
                      omega1=w1, omega2=w2,
                      R_star=R_star, I_star=I_star,
                      soop1_result=soop1_result, soop2_result=soop2_result,
                      full_history=full_history)


# ----------------------------------------------------------------------
# Metric evaluation
# ----------------------------------------------------------------------
def eval_metrics(scenario, sys_cfg, F, W):
    """Return (R, I) in bits/Hz using the existing sum_rate / sensing_mi."""
    R = sum_rate(scenario.H, F, W, sys_cfg.sigma2, scenario.Kc)
    I = sensing_mi(scenario.Bt_s, scenario.gamma_s2, F, W,
                    sys_cfg.sigma_s2, sys_cfg.L, scenario.Mr)
    return R, I


# ----------------------------------------------------------------------
# Result IO
# ----------------------------------------------------------------------
def save_result(name: str, result_dict: dict):
    """Save ``result_dict`` to ``result_v2/<name>.pkl`` (+ metadata json)."""
    RESULT_DIR.mkdir(exist_ok=True)
    pkl_path = RESULT_DIR / f"{name}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(result_dict, f)

    meta = {
        "exp": result_dict.get("exp"),
        "schemes": result_dict.get("schemes"),
        "x_label": result_dict.get("x_label"),
        "meta": result_dict.get("meta"),
    }
    json_path = RESULT_DIR / f"{name}.json"
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"[save_result] '{name}' -> {pkl_path}")


def load_result(name: str) -> dict:
    """Load ``result_v2/<name>.pkl``."""
    pkl_path = RESULT_DIR / f"{name}.pkl"
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def result_exists(name: str) -> bool:
    return (RESULT_DIR / f"{name}.pkl").exists()


def timestamp() -> str:
    """Human-readable UTC-local timestamp for ``meta['timestamp']``."""
    return time.strftime("%Y-%m-%d %H:%M:%S")
