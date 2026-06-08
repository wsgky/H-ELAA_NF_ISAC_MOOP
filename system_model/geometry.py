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
    Position of the N feeds u_i (3D), all on the array plane (x = 0).

    Parameters
    ----------
    structure : {"fully_connected", "subarray"}
        fully_connected : N feeds on a 2-D uniform grid spanning the full
                          aperture; aspect ratio N_h_f × N_v_f is chosen
                          to best match M_h / M_v.
        subarray        : each feed at the centroid of its M/N-element block.
    """
    V = build_array_positions(M_h, M_v, delta)  # (M, 3)
    M = M_h * M_v
    if structure == "fully_connected":
        # Factor N into (N_h_f, N_v_f) with aspect ratio closest to M_h/M_v.
        best = None
        for n_h in range(1, N + 1):
            if N % n_h == 0:
                n_v = N // n_h
                score = abs(n_h / n_v - M_h / M_v)
                if best is None or score < best[0]:
                    best = (score, n_h, n_v)
        N_h_f, N_v_f = best[1], best[2]
        ys = np.linspace(V[:, 1].min(), V[:, 1].max(), N_h_f)
        zs = np.linspace(V[:, 2].min(), V[:, 2].max(), N_v_f)
        # Ordering: n_v outer, n_h inner — consistent with build_array_positions.
        NV, NH = np.meshgrid(np.arange(N_v_f), np.arange(N_h_f), indexing="ij")
        y_feed = ys[NH.flatten()]
        z_feed = zs[NV.flatten()]
        return np.stack([np.zeros(N), y_feed, z_feed], axis=1)
    elif structure == "subarray":
        if M % N != 0:
            raise ValueError(f"For sub-array structure, N={N} must divide Mt={M}.")
        Msub = M // N
        U = np.zeros((N, 3))
        for i in range(N):
            block = V[i * Msub:(i + 1) * Msub]
            U[i] = block.mean(axis=0)  # x = 0 naturally since all elements have x = 0
        return U
    else:
        raise ValueError(f"Unknown structure: {structure}")


def random_near_field_position(rng: np.random.Generator,
                               r_range: Tuple[float, float],
                            #    half_space: bool = True
                               ) -> np.ndarray:
    """
    Draw a position (in 3D) inside the near-field region of the BS.

    The BS aperture is on the y-z plane around the origin and radiates
    towards +x, so we restrict azimuth phi to (-pi/2, pi/2) when
    half_space=True.
    """
    r = rng.uniform(*r_range)
    # print(f"Random near-field range: r={r:.2f} m")  # DEBUG
    # if half_space:
    #     phi = rng.uniform(-np.pi / 3 , np.pi / 3)
    # else:
    #     phi = rng.uniform(-np.pi, np.pi)
    phi = rng.uniform(-np.pi / 3 , np.pi / 3)
    theta = rng.uniform(-np.pi / 6, np.pi / 6)
    e = np.array([np.cos(theta) * np.cos(phi),
                  np.cos(theta) * np.sin(phi),
                  np.sin(theta)])
    return r * e, r, phi, theta
