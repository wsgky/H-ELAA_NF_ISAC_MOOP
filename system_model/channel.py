"""
Near-field array response & channels.

Implements Eqs. (11)-(19) of the manuscript:

For an array on the y-z plane with element positions
    v_m = [0, m_h * delta_h, m_v * delta_v]^T
and a scatter located at r_l = r_l * (cos(theta) cos(phi),
                                       cos(theta) sin(phi),
                                       sin(theta))^T,
the near-field array response is

    b(r, phi, theta) = b_F(phi, theta) ⊙ q(r, phi, theta),

with
    [b_F]_m = exp( j * kappa * (m_h * delta_h * cos(theta) sin(phi)
                                + m_v * delta_v * sin(theta)) )
    [q]_m   = exp( -j * kappa * q_m / (2 r) )
    q_m     = (m_h delta_h)^2 (1 - cos^2(theta) sin^2(phi))
            + (m_v delta_v)^2 cos^2(theta)
            - 2 m_h delta_h m_v delta_v cos(theta) sin(phi) sin(theta)

The CU channel h_k follows Eq. (11) (LoS + 1/sqrt(J) * sum NLoS).
The ST/CU/EO target response uses B(r, phi, theta) = b_r * b_t^H (Eq. 12).
"""
import numpy as np
from typing import Tuple
from .geometry import random_near_field_position


def near_field_array_response(M_h: int, M_v: int, delta: float,
                              r: float, phi: float, theta: float,
                              lambda0: float) -> np.ndarray:
    """
    Compute b(r, phi, theta) of length M = M_h * M_v.
    """
    kappa = 2 * np.pi / lambda0
    mh = np.arange(M_h)
    mv = np.arange(M_v)
    MV, MH = np.meshgrid(mv, mh, indexing="ij")  # (M_v, M_h)
    mh_grid = MH.flatten() * delta   # m_h * delta_h
    mv_grid = MV.flatten() * delta   # m_v * delta_v

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    sin_p = np.sin(phi)

    # far-field part b_F
    phase_far = kappa * (mh_grid * cos_t * sin_p + mv_grid * sin_t)
    b_F = np.exp(1j * phase_far)

    # distance-dependent part q
    qm = (mh_grid ** 2) * (1.0 - (cos_t * sin_p) ** 2) \
        + (mv_grid ** 2) * (cos_t ** 2) \
        - 2.0 * mh_grid * mv_grid * cos_t * sin_p * sin_t
    q = np.exp(-1j * kappa * qm / (2.0 * r))

    return b_F * q   # (M,)


# ----------------------------------------------------------------------
# CU channel (Eq. 11)
# ----------------------------------------------------------------------
def generate_channel_vector(rng: np.random.Generator,
                            M_h: int, M_v: int, delta: float, lambda0: float,
                            r_range: Tuple[float, float],
                            J: int,
                            los_gain_dB: float,
                            nlos_gain_dB: float,
                            path_loss_exp: float = 2.0,
                            position: np.ndarray = None,
                            ) -> Tuple[np.ndarray, dict]:
    """
    Generate one CU channel h_k of length Mt and return associated geometry.
    """
    Mt = M_h * M_v
    if position is None:
        pos, r0, phi0, theta0 = random_near_field_position(rng, r_range)
    else:
        pos = position
        r0 = np.linalg.norm(pos)
        phi0 = np.arctan2(pos[1], pos[0])
        theta0 = np.arcsin(pos[2] / max(r0, 1e-12))

    los_gain_lin = 10 ** (los_gain_dB / 20)
    alpha0 = los_gain_lin * (r0 ** (-path_loss_exp / 2)) \
        * (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)

    b0 = near_field_array_response(M_h, M_v, delta, r0, phi0, theta0, lambda0)
    h = alpha0 * b0

    # NLoS components, scatters around the user
    nlos_gain_lin = 10 ** (nlos_gain_dB / 20)
    info = {"r0": r0, "phi0": phi0, "theta0": theta0, "pos": pos,
            "alpha0": alpha0, "nlos_dirs": []}
    for j in range(J):
        # nearby scatter (random offset)
        offset = rng.normal(scale=1.0, size=3)
        scatter_pos = pos + offset
        rj = np.linalg.norm(scatter_pos)
        rj = max(rj, 0.5)
        phij = np.arctan2(scatter_pos[1], scatter_pos[0])
        thetaj = np.arcsin(scatter_pos[2] / rj)
        alphaj = nlos_gain_lin * (rj ** (-path_loss_exp / 2)) \
            * (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
        bj = near_field_array_response(M_h, M_v, delta, rj, phij, thetaj, lambda0)
        h = h + alphaj * bj / np.sqrt(J)
        info["nlos_dirs"].append((rj, phij, thetaj, alphaj))

    return h.reshape(1, Mt), info  # row vector (1, Mt) following manuscript


# ----------------------------------------------------------------------
# Target response (Eq. 12)
# ----------------------------------------------------------------------
def generate_target_response(rng: np.random.Generator,
                             Mt_h: int, Mt_v: int,
                             Mr_h: int, Mr_v: int,
                             delta: float, lambda0: float,
                             positions: list,
                             rcs_dB: float,
                             path_loss_exp: float = 2.0,
                             ) -> Tuple[np.ndarray, list, list, list]:
    """
    Build G = sum_i beta_i * b_r(r_i, phi_i, theta_i) * b_t^H(r_i, phi_i, theta_i)
    of size (Mr, Mt).

    Returns
    -------
    G : (Mr, Mt) complex
    Bt_list : list of b_t vectors (length Mt)
    Br_list : list of b_r vectors (length Mr)
    gammas2 : list of E[|beta_i|^2] used for sensing MI
    """
    Mt = Mt_h * Mt_v
    Mr = Mr_h * Mr_v
    G = np.zeros((Mr, Mt), dtype=complex)
    Bt_list, Br_list, gammas2 = [], [], []

    rcs_lin = 10 ** (rcs_dB / 20)

    for pos in positions:
        r = np.linalg.norm(pos)
        phi = np.arctan2(pos[1], pos[0])
        theta = np.arcsin(pos[2] / max(r, 1e-12))
        # complex reflectivity, CN(0, gamma_i^2)
        gamma2 = (rcs_lin * (r ** (-path_loss_exp))) ** 2
        beta = np.sqrt(gamma2) * (rng.standard_normal()
                                  + 1j * rng.standard_normal()) / np.sqrt(2)
        bt = near_field_array_response(Mt_h, Mt_v, delta, r, phi, theta, lambda0)
        br = near_field_array_response(Mr_h, Mr_v, delta, r, phi, theta, lambda0)
        G += beta * np.outer(br, bt.conj())
        Bt_list.append(bt)
        Br_list.append(br)
        gammas2.append(gamma2)

    return G, Bt_list, Br_list, gammas2
