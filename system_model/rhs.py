"""
Reconfigurable Holographic Surface (RHS) related quantities.

Implements:
- Phi_{m,i} = exp(-j * kappa * || d_{m,i} ||)            (Eq. 1)
  with d_{m,i} = u_i - v_m
- Holographic beamforming matrix F = A * Phi
  * Fully-connected:  Phi = [Phi_1, ..., Phi_N], col i over all M_t
                       elements, normalised by 1/sqrt(M_t).            (Eq. 2)
  * Sub-array:         Phi is block-diagonal, each block a length-M
                       vector normalised by 1/sqrt(M).                  (Eq. 3)
- A = diag(a), with a_m in [0, 1].
"""
import numpy as np
from .geometry import build_array_positions, feed_positions


def _phase_response(V: np.ndarray, U: np.ndarray, kappa: float) -> np.ndarray:
    """
    Returns the M-by-N phase response matrix whose (m,i) entry is
        exp(-j * kappa * || u_i - v_m ||).
    """
    # diff[m, i, :] = u_i - v_m
    diff = U[None, :, :] - V[:, None, :]
    dist = np.linalg.norm(diff, axis=2)  # (M, N)
    return np.exp(-1j * kappa * dist)


def build_phase_matrix(M_h: int, M_v: int, N: int, delta: float,
                       lambda_w: float, structure: str = "subarray"
                       ) -> np.ndarray:
    """
    Build the (Mt, N) phase matrix Phi.

    Parameters
    ----------
    M_h, M_v : int
        Horizontal / vertical element counts of the surface.
    N : int
        Number of feeds.
    delta : float
        Element spacing (also used to place the feeds).
    lambda_w : float
        Wavelength inside the waveguide. kappa = 2*pi/lambda_w.
    structure : {"fully_connected", "subarray"}

    Returns
    -------
    Phi : (Mt, N) complex ndarray
    """
    V = build_array_positions(M_h, M_v, delta)
    U = feed_positions(N, M_h, M_v, delta, structure=structure)
    kappa = 2 * np.pi / lambda_w
    Mt = M_h * M_v

    if structure == "fully_connected":
        Phi_full = _phase_response(V, U, kappa)    # (Mt, N)
        Phi = Phi_full / np.sqrt(Mt)
        return Phi

    if structure == "subarray":
        if Mt % N != 0:
            raise ValueError(f"N={N} must divide Mt={Mt} for sub-array structure.")
        Msub = Mt // N
        Phi = np.zeros((Mt, N), dtype=complex)
        scale = 1.0 / np.sqrt(Msub)
        for i in range(N):
            block_idx = np.arange(i * Msub, (i + 1) * Msub)
            Vi = V[block_idx]
            ui = U[i:i + 1]
            phi_block = _phase_response(Vi, ui, kappa).flatten()  # (Msub,)
            Phi[block_idx, i] = phi_block * scale
        return Phi

    raise ValueError(f"Unknown structure: {structure}")


def holographic_beamforming_matrix(a: np.ndarray, Phi: np.ndarray) -> np.ndarray:
    """F = diag(a) @ Phi.  a: (Mt,) real in [0, 1];  Phi: (Mt, N)."""
    return a[:, None] * Phi
