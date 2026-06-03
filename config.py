"""
Global configuration for the Holographic-ELAA ISAC simulation.

All default parameters are gathered here so that experiments can override
specific entries without touching the rest of the pipeline.

Notation follows the manuscript:
    Mt, Mr   : number of TX/RX RHS elements
    N        : number of RF chains / feeds
    Kc       : number of communication users (CU)
    Ks       : number of sensing targets (ST)
    Ke       : number of environmental objects (EO, clutter)
    L        : symbol block length
    Pt       : total transmit power budget
    delta    : antenna element spacing
    lambda0  : carrier wavelength in free space
    lambda_w : wavelength in the waveguide (kappa = 2*pi/lambda_w)
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SystemConfig:
    # ---- RHS / array geometry ----
    Mt_h: int = 32                  # horizontal elements (TX RHS)
    Mt_v: int = 32                  # vertical elements (TX RHS)
    Mr_h: int = 32                  # horizontal elements (RX RHS)
    Mr_v: int = 32                  # vertical elements (RX RHS)
    N: int = 8                      # number of RF chains / feeds
    rhs_structure: str = "subarray"  # "fully_connected" or "subarray"

    # ---- carrier / wavelength ----
    fc: float = 30e9                # carrier frequency (Hz), mmWave
    eps_r: float = 3.0              # relative permittivity inside waveguide
    delta_factor: float = 0.25       # delta = delta_factor * lambda0

    # ---- users / targets ----
    Kc: int = 3                     # # CUs
    Ks: int = 2                     # # STs
    Ke: int = 2                     # # EOs (clutter)
    J_nlos: int = 3                 # # NLoS scatters per CU
    near_field_radius: Tuple[float, float] = (2.0, 15.0)  # meters

    # ---- power & noise ----
    Pt_dBm: float = 10.0            # total TX power (dBm)
    sigma2_dBm: float = 0.0       # noise power per CU (dBm)
    sigma_s2_dBm: float = 0.0     # sensing noise (dBm)
    L: int = 256                    # symbol block length

    # ---- path-gain / RCS ----
    kappa: float = 10.0             # Rician factor (LoS/NLoS power ratio)
    rcs_st_dB: float = 0.0         # ST gamma^2 variance in dB (10^(x/10))
    rcs_cu_dB: float = 0.0         # CU clutter gamma^2 variance in dB
    rcs_eo_dB: float = 0.0          # EO gamma^2 variance in dB

    # ---- random seed ----
    seed: int = 2025

    # -------- derived properties --------
    @property
    def Mt(self) -> int:
        return self.Mt_h * self.Mt_v

    @property
    def Mr(self) -> int:
        return self.Mr_h * self.Mr_v

    @property
    def lambda0(self) -> float:
        c = 3e8
        return c / self.fc

    @property
    def lambda_w(self) -> float:
        # wavelength inside dielectric waveguide
        import math
        return self.lambda0 / math.sqrt(self.eps_r)

    @property
    def delta(self) -> float:
        return self.delta_factor * self.lambda0

    @property
    def Pt(self) -> float:
        return 10 ** (self.Pt_dBm / 10) * 1e-3

    @property
    def sigma2(self) -> float:
        return 10 ** (self.sigma2_dBm / 10) * 1e-3

    @property
    def sigma_s2(self) -> float:
        return 10 ** (self.sigma_s2_dBm / 10) * 1e-3


@dataclass
class AlgorithmConfig:
    # ---- outer / inner iterations ----
    SOOP1_outer_iters: int = 20
    SOOP2_outer_iters: int = 20
    MOOP_outer_iters: int = 10
    SOOP1_inner_iters: int = 100
    SOOP2_inner_iters: int = 100
    MOOP_inner_iters: int = 500

    sp5_iters: int = 1       # max SCA iterations inside SP5 at outer iter 0
    sp5_decay: float = 1.0   # harmonic decay: n_sca(s) = max(1, round(sp5_iters/(1+decay*s)))
    sp5_tol: float = 1e-3    # relative ΔW early-stop threshold within SP5 SCA loop
    pgd_steps: int = 20      # PGD steps per surrogate refresh (SOOP1 SP2)
    tol: float = 1e-5

    # ---- step sizes ----
    pgd_step_a: float = 1e-2        # for amplitude PGD (SOOP2 / MOOP)
    pgd_step_lambda: float = 1e-2   # for dual update (MOOP)

    # ---- MOOP SP6 monotonic backtracking line search ----
    # The amplitude (a) update is a projected-gradient ascent step on the
    # Lagrangian. A plain step does NOT guarantee the *original* Tchebycheff
    # objective tau rises, so we accept a candidate only if it does not
    # decrease tau; otherwise the step is shrunk by bt_beta and retried.
    bt_beta: float = 0.5            # step shrink factor (0<beta<1)
    bt_max: int = 20                # max backtracking trials per inner step
    MOOP_inner_patience: int = 15   # stop SP6 after this many no-progress steps
    MOOP_outer_tol: float = 5e-4   # min τ improvement per outer iter to count as progress
    MOOP_outer_patience: int = 3   # stop outer BCD after this many consecutive no-progress iters

    # ---- weighting (Tchebycheff) ----
    omega1: float = 0.5
    omega2: float = 0.5

    # ---- CVX backend ----
    cvx_solver: str = "SCS"         # 'SCS' / 'CLARABEL' / 'MOSEK'
    cvx_verbose: bool = False

    # ---- numerical guards ----
    eig_tol: float = 1e-9
    nesterov: bool = True
