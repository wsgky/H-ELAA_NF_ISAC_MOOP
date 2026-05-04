"""
Common utilities shared by all optimisation algorithms:
- Sum-rate / SINR computation for the CUs (Eq. 7-8).
- Asymptotic sensing mutual information (Eq. 35).
- Power scaling, projections, etc.

These are deliberately pure functions of (W, A, scenario): if either the
scenario generator or the algorithm has a bug, the numbers produced
here can be checked independently.
"""
import numpy as np


# ----------------------------------------------------------------------
# Communication metrics
# ----------------------------------------------------------------------
def compute_F(a: np.ndarray, Phi: np.ndarray) -> np.ndarray:
    return a[:, None] * Phi


def compute_sinr(H: np.ndarray, F: np.ndarray, W: np.ndarray,
                 sigma2: float, Kc: int) -> np.ndarray:
    """
    SINR_k for the Kc CUs (Eq. 8).

    H : (Kc, Mt)
    F : (Mt, N)
    W : (N, Kc + Ks)  -- columns 0..Kc-1 communication, rest sensing
    """
    HF = H @ F                       # (Kc, N)
    sig = HF @ W                     # (Kc, Kc+Ks)
    abs2 = np.abs(sig) ** 2          # (Kc, Kc+Ks)
    sinrs = np.zeros(Kc)
    for k in range(Kc):
        num = abs2[k, k]
        # interference from other CUs and sensing streams
        denom = abs2[k, :].sum() - abs2[k, k] + sigma2
        sinrs[k] = num / max(denom, 1e-30)
    return sinrs


def sum_rate(H: np.ndarray, F: np.ndarray, W: np.ndarray,
             sigma2: float, Kc: int) -> float:
    sinrs = compute_sinr(H, F, W, sigma2, Kc)
    return float(np.sum(np.log2(1.0 + sinrs)))


# ----------------------------------------------------------------------
# Sensing mutual information (asymptotic, Eq. 35)
# ----------------------------------------------------------------------
def sensing_mi(Bt_s: np.ndarray, gamma_s2: np.ndarray,
               F: np.ndarray, W: np.ndarray,
               sigma_s2: float, L: int, Mr: int) -> float:
    """
    I = sum_i log2(1 + (L * Mr * gamma_i^2 / sigma^2) * b_i^H F W W^H F^H b_i).

    Parameters
    ----------
    Bt_s : (Mt, Ks) transmit array responses for the STs
    gamma_s2 : (Ks,) E[|beta_i|^2]
    """
    FW = F @ W                            # (Mt, Kc+Ks)
    FWWF = FW @ FW.conj().T               # (Mt, Mt)
    val = 0.0
    for i in range(Bt_s.shape[1]):
        b = Bt_s[:, i]
        gain = np.real(b.conj() @ FWWF @ b)
        snr_i = (L * Mr * gamma_s2[i] / sigma_s2) * gain
        val += np.log2(1.0 + max(snr_i, 0.0))
    return float(val)


# ----------------------------------------------------------------------
# Power-related helpers
# ----------------------------------------------------------------------
def transmit_power(F: np.ndarray, W: np.ndarray) -> float:
    """Tr(F W W^H F^H)."""
    FW = F @ W
    return float(np.real(np.trace(FW @ FW.conj().T)))


def scale_W_to_power(F: np.ndarray, W: np.ndarray, Pt: float) -> np.ndarray:
    """If transmit_power(F, W) > Pt, scale W down to meet the budget."""
    p = transmit_power(F, W)
    if p <= Pt or p <= 0:
        return W
    return W * np.sqrt(Pt / p)


def project_box(x: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    return np.clip(x, lo, hi)
