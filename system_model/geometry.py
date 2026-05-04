"""
Array / feed geometry.

The RHS lies in the y-z plane (x = 0). For the m-th element at index
(m_h, m_v):
        v_m = [0, m_h * delta, m_v * delta]^T
"""
import numpy as np
from typing import Tuple


def build_array_positions(M_h: int, M_v: int, delta: float) -> np.ndarray:
    """
    Element positions of an RHS / UPA on the y-z plane.

    Returns
    -------
    V : (M, 3) ndarray, with M = M_h * M_v
        Row m is [0, m_h*delta, m_v*delta]. Order: (m_v outer, m_h inner)
        i.e. m = m_v * M_h + m_h, matching the manuscript convention
        v_m = (0, m_h*delta, m_v*delta).
    """
    mh = np.arange(M_h)
    mv = np.arange(M_v)
    MV, MH = np.meshgrid(mv, mh, indexing="ij")  # (M_v, M_h)
    y = MH.flatten() * delta
    z = MV.flatten() * delta
    x = np.zeros_like(y)
    return np.stack([x, y, z], axis=1)  # (M, 3)


def feed_positions(N: int, M_h: int, M_v: int, delta: float,
                   structure: str = "subarray") -> np.ndarray:
    """
    Position of the N feeds u_i (3D).

    For both structures the feeds are placed slightly behind the surface
    (x = -delta) and roughly aligned with the centroids of their
    associated element groups, so that the phase matrix Phi is well
    behaved.

    Parameters
    ----------
    structure : {"fully_connected", "subarray"}
    """
    V = build_array_positions(M_h, M_v, delta)  # (M, 3)
    M = M_h * M_v
    if structure == "fully_connected":
        # Place N feeds along a horizontal line, slightly behind the surface
        # and roughly spanning the array's vertical centre.
        ys = np.linspace(V[:, 1].min(), V[:, 1].max(), N)
        z_mid = 0.5 * (V[:, 2].min() + V[:, 2].max())
        U = np.stack([-delta * np.ones(N), ys, z_mid * np.ones(N)], axis=1)
        return U
    elif structure == "subarray":
        if M % N != 0:
            raise ValueError(f"For sub-array structure, N={N} must divide Mt={M}.")
        Msub = M // N
        # The i-th feed is centred on the i-th block of M_sub elements.
        U = np.zeros((N, 3))
        for i in range(N):
            block = V[i * Msub:(i + 1) * Msub]
            U[i] = block.mean(axis=0)
            U[i, 0] = -delta  # slightly behind
        return U
    else:
        raise ValueError(f"Unknown structure: {structure}")


def random_near_field_position(rng: np.random.Generator,
                               r_range: Tuple[float, float],
                               half_space: bool = True) -> np.ndarray:
    """
    Draw a position (in 3D) inside the near-field region of the BS.

    The BS aperture is on the y-z plane around the origin and radiates
    towards +x, so we restrict azimuth phi to (-pi/2, pi/2) when
    half_space=True.
    """
    r = rng.uniform(*r_range)
    if half_space:
        phi = rng.uniform(-np.pi / 2 + 0.05, np.pi / 2 - 0.05)
    else:
        phi = rng.uniform(-np.pi, np.pi)
    theta = rng.uniform(-np.pi / 4, np.pi / 4)
    e = np.array([np.cos(theta) * np.cos(phi),
                  np.cos(theta) * np.sin(phi),
                  np.sin(theta)])
    return r * e, r, phi, theta
