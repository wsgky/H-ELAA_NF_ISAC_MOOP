"""
config_v2.py -- parameters for the Numerical-Results Simulation Suite (v2).

This module is purely additive: it reuses the existing, read-only
``config.SystemConfig`` / ``config.AlgorithmConfig`` dataclasses for every
per-run parameter, and only adds:

  * ``base_sys_cfg()`` / ``base_alg_cfg()`` -- Table I defaults.
  * Sweep grids / weight sets / Monte-Carlo settings shared by
    ``experiments_v2`` (single source of truth for "what gets swept").
  * ``ExperimentConfig`` + ``get_config(smoke=False)`` -- one object,
    built from the above, that every ``experiments_v2/exp_*.py::run(cfg)``
    consumes.
  * ``full_trajectory_alg_cfg(alg_cfg)`` -- disables the MOOP outer-loop
    early-stop patience (sets ``MOOP_outer_patience = MOOP_outer_iters``)
    so ``exp_convergence`` / ``exp_iteration_cdf`` see the full
    Smax-length tau trajectory instead of an early-stopped one.

Design notes / intentional deviations from a literal Table I reading
----------------------------------------------------------------------
- ``delta = lambda0 / 2``  -> ``delta_factor = 0.5``.
- ``lambda_g = lambda_0``  -> ``eps_r = 1.0`` (since
  ``lambda_w = lambda0 / sqrt(eps_r)``).
- ``N`` is FIXED at 8 for every experiment (not ``Kc + Ks``). The default
  "subarray" RHS structure (``system_model.rhs.build_phase_matrix``)
  requires ``N | Mt``. With ``Mt_v = 16`` (resp. 8 in --smoke) fixed
  across every sweep (``exp_vs_mt``, ``exp_vs_k``) and ``8 | 16``
  (resp. ``8 | 8``), ``N = 8`` divides ``Mt = Mt_h * Mt_v`` for *every*
  ``Mt_h``, and divides ``Mt = 2048`` (resp. 128) for the Kc/Ks sweeps.
  ``Kc + Ks`` (e.g. 6, 7, 10, ...) generally does NOT divide
  ``Mt = Mt_h * 16`` and would raise inside ``build_phase_matrix``.
"""
import copy
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from config import SystemConfig, AlgorithmConfig


# ----------------------------------------------------------------------
# Table I defaults
# ----------------------------------------------------------------------
SMAX = 100          # Algorithm 1 outer-iteration cap (manuscript Smax)
TMAX = 500         # inner-loop iteration cap (manuscript Tmax)
EPS_TAU = 1e-4     # manuscript epsilon: |tau^(s) - tau^(s-1)| <= eps

N_MC = 5          # Monte Carlo trials (Sec. 4: "50 for all experiments")
BASE_SEED = 2025
FIXED_W1 = 0.5     # fixed omega1 used by exp_vs_power / exp_vs_mt / exp_vs_k

WEIGHTS_PARETO = np.linspace(0.0, 1.0, 21)
# WEIGHTS_CONV = [0.2, 0.5, 0.8]
WEIGHTS_CONV = [0.2]
PT_GRID_DBM = [10, 15, 20, 25, 30, 35, 40]

MT_V_FIXED = 16
MT_H_GRID = [32, 64, 96, 128]

KS_FIXED_FOR_KC_SWEEP = 2
KC_GRID = [2, 3, 4, 5, 6]

KC_FIXED_FOR_KS_SWEEP = 4
KS_GRID = [1, 2, 3, 4]

# --- smoke-mode (fast, reduced) overrides ---
N_MC_SMOKE = 3
SMAX_SMOKE = 10
TMAX_SMOKE = 20
MT_H_SMOKE, MT_V_SMOKE = 16, 8     # Mt = 128, still 8 | Mt
PT_GRID_DBM_SMOKE = [20, 30]
MT_H_GRID_SMOKE = [16, 32]
KC_GRID_SMOKE = [2, 3]
KS_GRID_SMOKE = [1, 2]

# Must match experiments_v2.schemes.SCHEME_NAMES (kept as a literal here
# to avoid experiments_v2 <-> config_v2 import-order coupling).
# DEFAULT_SCHEMES = [
#     "proposed",
#     "fully_digital",
#     "amplitude_only",
#     "random_rhs",
#     "uniform_rhs",
#     "far_field",
# ]
DEFAULT_SCHEMES = [
     "proposed"]

# ----------------------------------------------------------------------
# Base SystemConfig / AlgorithmConfig (Table I)
# ----------------------------------------------------------------------
def base_sys_cfg() -> SystemConfig:
    return SystemConfig(
        Mt_h=128, Mt_v=MT_V_FIXED,   # Mt = 2048
        Mr_h=128, Mr_v=MT_V_FIXED,   # Mr = 2048
        N=8,
        rhs_structure="fully_connected",  # "subarray" also possible but not used in paper
        fc=30e9,
        eps_r=1.0,            # lambda_g = lambda_0
        delta_factor=0.5,     # delta = lambda_0 / 2
        Kc=4, Ks=2, Ke=2,
        J_nlos=5,
        L=1024,
        Pt_dBm=30.0,
        seed=BASE_SEED,
    )


def base_alg_cfg() -> AlgorithmConfig:
    return AlgorithmConfig(
        SOOP1_outer_iters=SMAX,
        SOOP2_outer_iters=SMAX,
        MOOP_outer_iters=SMAX,
        SOOP1_inner_iters=TMAX,
        SOOP2_inner_iters=TMAX,
        MOOP_inner_iters=TMAX,
        tol=EPS_TAU,
        MOOP_outer_tol=EPS_TAU,
    )


# ----------------------------------------------------------------------
# Experiment configuration bundle
# ----------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    sys_cfg: SystemConfig
    alg_cfg: AlgorithmConfig
    smoke: bool = False

    n_mc: int = N_MC
    base_seed: int = BASE_SEED
    fixed_w1: float = FIXED_W1
    schemes: Optional[List[str]] = None

    Smax: int = SMAX
    eps_tau: float = EPS_TAU

    weights_pareto: np.ndarray = field(default_factory=lambda: WEIGHTS_PARETO)
    weights_conv: List[float] = field(default_factory=lambda: list(WEIGHTS_CONV))

    Pt_grid_dBm: List[float] = field(default_factory=lambda: list(PT_GRID_DBM))

    Mt_h_grid: List[int] = field(default_factory=lambda: list(MT_H_GRID))
    Mt_v_fixed: int = MT_V_FIXED

    Kc_grid: List[int] = field(default_factory=lambda: list(KC_GRID))
    Ks_fixed_for_kc: int = KS_FIXED_FOR_KC_SWEEP
    Ks_grid: List[int] = field(default_factory=lambda: list(KS_GRID))
    Kc_fixed_for_ks: int = KC_FIXED_FOR_KS_SWEEP

    def scheme_list(self) -> List[str]:
        return list(self.schemes) if self.schemes is not None else list(DEFAULT_SCHEMES)


def get_config(smoke: bool = False) -> ExperimentConfig:
    """Build the shared ``ExperimentConfig`` used by every experiment.

    With ``smoke=True``: smaller array, fewer outer/inner iterations,
    n_mc=3, and coarser sweep grids -- for ``python main_v2.py --smoke``.
    """
    sys_cfg = base_sys_cfg()
    alg_cfg = base_alg_cfg()

    if not smoke:
        return ExperimentConfig(sys_cfg=sys_cfg, alg_cfg=alg_cfg, smoke=False)

    sys_cfg = copy.deepcopy(sys_cfg)
    sys_cfg.Mt_h, sys_cfg.Mt_v = MT_H_SMOKE, MT_V_SMOKE
    sys_cfg.Mr_h, sys_cfg.Mr_v = MT_H_SMOKE, MT_V_SMOKE
    sys_cfg.L = 64

    alg_cfg = copy.deepcopy(alg_cfg)
    alg_cfg.SOOP1_outer_iters = SMAX_SMOKE
    alg_cfg.SOOP2_outer_iters = SMAX_SMOKE
    alg_cfg.MOOP_outer_iters = SMAX_SMOKE
    alg_cfg.SOOP1_inner_iters = TMAX_SMOKE
    alg_cfg.SOOP2_inner_iters = TMAX_SMOKE
    alg_cfg.MOOP_inner_iters = TMAX_SMOKE

    return ExperimentConfig(
        sys_cfg=sys_cfg, alg_cfg=alg_cfg, smoke=True,
        n_mc=N_MC_SMOKE,
        Smax=SMAX_SMOKE,
        weights_pareto=np.linspace(0.0, 1.0, 5),
        Pt_grid_dBm=list(PT_GRID_DBM_SMOKE),
        Mt_h_grid=list(MT_H_GRID_SMOKE),
        Mt_v_fixed=MT_V_SMOKE,
        Kc_grid=list(KC_GRID_SMOKE),
        Ks_grid=list(KS_GRID_SMOKE),
    )


def full_trajectory_alg_cfg(alg_cfg: AlgorithmConfig) -> AlgorithmConfig:
    """Return a copy of ``alg_cfg`` with MOOP outer-loop early-stop disabled.

    ``exp_convergence`` / ``exp_iteration_cdf`` need the full
    ``MOOP_outer_iters``-length tau trajectory; the default
    ``MOOP_outer_patience`` would otherwise truncate it once tau plateaus.
    """
    cfg = copy.deepcopy(alg_cfg)
    cfg.MOOP_outer_patience = cfg.MOOP_outer_iters
    return cfg
