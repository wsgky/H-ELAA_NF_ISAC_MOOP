"""
Scenario object: wraps one full ISAC instance.

A scenario contains:
- The TX/RX RHS phase matrices Phi (transmit) and Phi_r (not used directly,
  but kept for completeness).
- CU channels h_k (Kc x Mt), positions, NLoS info.
- ST positions and the corresponding b_t, b_r vectors and gamma_s^2.
- CU positions, b_t, b_r as clutter (already implicit from h_k LoS, but we
  also need CU-side b_t for the sensing model in (6)).
- EO positions and the corresponding b_t, b_r and gamma_e^2.

The optimisation algorithms only see the *outputs* of this object, hence
system-model bugs and algorithm bugs can be debugged independently.
"""
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

from .rhs import build_phase_matrix
from .channel import (
    generate_channel_vector,
    generate_target_response,
    near_field_array_response,
)
from .geometry import random_near_field_position


@dataclass
class Scenario:
    # geometry / RHS
    Mt_h: int
    Mt_v: int
    Mr_h: int
    Mr_v: int
    N: int
    delta: float
    lambda0: float
    structure: str

    # CU
    H: np.ndarray                    # (Kc, Mt) channel matrix
    cu_info: list                    # list of dicts with positions etc.

    # ST
    Bt_s: np.ndarray                 # (Mt, Ks) transmit array responses
    Br_s: np.ndarray                 # (Mr, Ks)
    gamma_s2: np.ndarray             # (Ks,)

    # CU as clutter (for sensing): we need their b_t and b_r
    Bt_c: np.ndarray                 # (Mt, Kc)
    Br_c: np.ndarray                 # (Mr, Kc)
    gamma_c2: np.ndarray             # (Kc,)

    # EO clutter
    Bt_e: np.ndarray                 # (Mt, Ke)
    Br_e: np.ndarray                 # (Mr, Ke)
    gamma_e2: np.ndarray             # (Ke,)

    # RHS phase matrix (TX)
    Phi: np.ndarray                  # (Mt, N)

    # raw positions, for plotting / debugging
    cu_positions: list = field(default_factory=list)
    st_positions: list = field(default_factory=list)
    eo_positions: list = field(default_factory=list)

    # ---- convenience ----
    @property
    def Mt(self) -> int:
        return self.Mt_h * self.Mt_v

    @property
    def Mr(self) -> int:
        return self.Mr_h * self.Mr_v

    @property
    def Kc(self) -> int:
        return self.H.shape[0]

    @property
    def Ks(self) -> int:
        return self.Bt_s.shape[1]

    @property
    def Ke(self) -> int:
        return self.Bt_e.shape[1]


def generate_scenario(cfg, rng: np.random.Generator = None) -> Scenario:
    """
    Generate a full random scenario from a SystemConfig.
    """
    if rng is None:
        rng = np.random.default_rng(cfg.seed)

    Phi = build_phase_matrix(cfg.Mt_h, cfg.Mt_v, cfg.N,
                             cfg.delta, cfg.lambda_w,
                             structure=cfg.rhs_structure)

    # ---------------- CUs ----------------
    Kc = cfg.Kc
    Mt = cfg.Mt
    H = np.zeros((Kc, Mt), dtype=complex)
    cu_info, cu_positions = [], []
    Bt_c = np.zeros((Mt, Kc), dtype=complex)
    Br_c = np.zeros((cfg.Mr, Kc), dtype=complex)
    gamma_c2 = np.zeros(Kc)

    for k in range(Kc):
        h_k, info = generate_channel_vector(
            rng, cfg.Mt_h, cfg.Mt_v, cfg.delta, cfg.lambda0,
            cfg.near_field_radius, cfg.J_nlos,
            cfg.los_gain_dB, cfg.nlos_gain_dB, cfg.path_loss_exp)
        H[k:k+1] = h_k
        cu_info.append(info)
        cu_positions.append(info["pos"])

        # CU also acts as clutter for sensing
        rk, pk, tk = info["r0"], info["phi0"], info["theta0"]
        Bt_c[:, k] = near_field_array_response(cfg.Mt_h, cfg.Mt_v, cfg.delta,
                                               rk, pk, tk, cfg.lambda0)
        Br_c[:, k] = near_field_array_response(cfg.Mr_h, cfg.Mr_v, cfg.delta,
                                               rk, pk, tk, cfg.lambda0)
        rcs_lin = 10 ** (cfg.rcs_cu_dB / 20)
        gamma_c2[k] = (rcs_lin * (rk ** (-cfg.path_loss_exp))) ** 2

    # ---------------- STs ----------------
    Ks = cfg.Ks
    st_positions = []
    Bt_s = np.zeros((Mt, Ks), dtype=complex)
    Br_s = np.zeros((cfg.Mr, Ks), dtype=complex)
    gamma_s2 = np.zeros(Ks)
    rcs_lin_s = 10 ** (cfg.rcs_st_dB / 20)
    for i in range(Ks):
        pos, r, phi, theta = random_near_field_position(rng, cfg.near_field_radius)
        st_positions.append(pos)
        Bt_s[:, i] = near_field_array_response(cfg.Mt_h, cfg.Mt_v, cfg.delta,
                                               r, phi, theta, cfg.lambda0)
        Br_s[:, i] = near_field_array_response(cfg.Mr_h, cfg.Mr_v, cfg.delta,
                                               r, phi, theta, cfg.lambda0)
        gamma_s2[i] = (rcs_lin_s * (r ** (-cfg.path_loss_exp))) ** 2

    # ---------------- EOs ----------------
    Ke = cfg.Ke
    eo_positions = []
    Bt_e = np.zeros((Mt, Ke), dtype=complex)
    Br_e = np.zeros((cfg.Mr, Ke), dtype=complex)
    gamma_e2 = np.zeros(Ke)
    rcs_lin_e = 10 ** (cfg.rcs_eo_dB / 20)
    for j in range(Ke):
        pos, r, phi, theta = random_near_field_position(rng, cfg.near_field_radius)
        eo_positions.append(pos)
        Bt_e[:, j] = near_field_array_response(cfg.Mt_h, cfg.Mt_v, cfg.delta,
                                               r, phi, theta, cfg.lambda0)
        Br_e[:, j] = near_field_array_response(cfg.Mr_h, cfg.Mr_v, cfg.delta,
                                               r, phi, theta, cfg.lambda0)
        gamma_e2[j] = (rcs_lin_e * (r ** (-cfg.path_loss_exp))) ** 2

    return Scenario(
        Mt_h=cfg.Mt_h, Mt_v=cfg.Mt_v, Mr_h=cfg.Mr_h, Mr_v=cfg.Mr_v,
        N=cfg.N, delta=cfg.delta, lambda0=cfg.lambda0,
        structure=cfg.rhs_structure,
        H=H, cu_info=cu_info,
        Bt_s=Bt_s, Br_s=Br_s, gamma_s2=gamma_s2,
        Bt_c=Bt_c, Br_c=Br_c, gamma_c2=gamma_c2,
        Bt_e=Bt_e, Br_e=Br_e, gamma_e2=gamma_e2,
        Phi=Phi,
        cu_positions=cu_positions, st_positions=st_positions,
        eo_positions=eo_positions,
    )
