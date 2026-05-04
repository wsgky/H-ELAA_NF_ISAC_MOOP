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
    delta_factor: float = 0.5       # delta = delta_factor * lambda0

    # ---- users / targets ----
    Kc: int = 3                     # # CUs
    Ks: int = 2                     # # STs
    Ke: int = 2                     # # EOs (clutter)
    J_nlos: int = 3                 # # NLoS scatters per CU
    near_field_radius: Tuple[float, float] = (2.0, 15.0)  # meters

    # ---- power & noise ----
    Pt_dBm: float = 30.0            # total TX power (dBm)
    sigma2_dBm: float = -90.0       # noise power per CU (dBm)
    sigma_s2_dBm: float = -90.0     # sensing noise (dBm)
    L: int = 256                    # symbol block length

    # ---- path-gain / RCS ----
    path_loss_exp: float = 2.0
    los_gain_dB: float = -60.0      # reference LoS gain at r=1m
    nlos_gain_dB: float = -75.0     # reference NLoS gain
    rcs_st_dB: float = 10.0         # ST reflectivity (dBsm scale, relative)
    rcs_cu_dB: float = -5.0         # CU reflectivity as clutter
    rcs_eo_dB: float = 0.0          # EO reflectivity

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
    outer_iters: int = 20
    inner_iters: int = 30
    tol: float = 1e-4

    # ---- step sizes ----
    pgd_step_a: float = 5e-3        # for amplitude PGD (SOOP2 / MOOP)
    pgd_step_lambda: float = 1e-2   # for dual update (MOOP)

    # ---- weighting (Tchebycheff) ----
    omega1: float = 0.5
    omega2: float = 0.5

    # ---- CVX backend ----
    cvx_solver: str = "SCS"         # 'SCS' / 'CLARABEL' / 'MOSEK'
    cvx_verbose: bool = False

    # ---- numerical guards ----
    eig_tol: float = 1e-9
    nesterov: bool = True
