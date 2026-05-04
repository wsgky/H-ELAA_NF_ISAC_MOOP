"""
Algorithm 1 (SOOP1): communication-centric beamforming.

Solves
    max  R = sum_k log(1 + SINR_k)
    s.t. Tr(F W W^H F^H) <= Pt,  0 <= a_m <= 1.

Pipeline (alternating optimisation, manuscript Sec. V-A):

  Outer iteration i:
    1) Build effective channel  bar{H}^(c) = H F^(i-1)
    2) Solve SP1: digital beamforming = zero-forcing + water-filling
    3) Solve SP2 (subproblem on a) with FP + accelerated non-homogeneous
       quadratic transform (Eq. 73-80), inner iteration t:
         - update auxiliary variables mu_k (Eq. 51) and xi_k (Eq. 54)
         - build  bar{Q} (Eq. 71),  bar{u} (Eq. 72)
         - lambda = ||bar{Q}||_F  (manuscript-suggested choice)
         - Nesterov extrapolation v^(t-1) (Eq. 79)
         - PGD step (Eq. 80) with box projection
"""
import numpy as np
from typing import Tuple, Dict, List

from .utils import (
    compute_F, compute_sinr, sum_rate, scale_W_to_power, project_box,
)


def _zf_waterfilling(H_eff: np.ndarray, sigma2: float, Pt: float
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zero-forcing precoder + water-filling power allocation.

    H_eff : (Kc, N).  Returns
        W_c : (N, Kc) zero-forced precoding matrix (already power-loaded)
        p   : (Kc,) per-stream power
    """
    Kc, N = H_eff.shape
    # pseudo-inverse  W_bar = H^H (H H^H)^{-1}    (since rows of H are channels)
    HHH = H_eff @ H_eff.conj().T
    HHH_inv = np.linalg.pinv(HHH)
    W_bar = H_eff.conj().T @ HHH_inv             # (N, Kc)

    norms2 = np.sum(np.abs(W_bar) ** 2, axis=0)  # (Kc,)
    norms2 = np.maximum(norms2, 1e-12)

    # Water-filling for max sum_k log(1 + p_k * sigma2 / norms2[k]) ?
    # Following Eq. (65), the per-stream "effective noise" is sigma_k^2 / ||w_bar_k||^2.
    # Water-filling solution: p_k = max(0, mu - sigma_k^2 / ||w_bar_k||^2).
    eff_noise = sigma2 / norms2
    # Bisection on mu so that sum p_k = Pt
    lo, hi = 0.0, np.max(eff_noise) + Pt + 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        p = np.maximum(0.0, mid - eff_noise)
        if p.sum() > Pt:
            hi = mid
        else:
            lo = mid
    p = np.maximum(0.0, mid - eff_noise)
    if p.sum() > 0:
        # rescale W_bar columns
        W_c = W_bar * np.sqrt(p)[None, :] / np.sqrt(norms2)[None, :]
        # Note: stream k transmits with power p_k. Since the precoder is
        # ||w_bar_k||*scaling, we set w_k = w_bar_k * sqrt(p_k / norms2_k).
    else:
        W_c = W_bar * 0.0
    return W_c, p


def _sp2_inner(a_init: np.ndarray, H: np.ndarray, Phi: np.ndarray,
               W: np.ndarray, sigma2: float, Kc: int,
               iters: int = 30, use_nesterov: bool = True
               ) -> Tuple[np.ndarray, list]:
    """
    Inner loop for SP2 (amplitude update) using FP + accelerated
    non-homogeneous quadratic transform.

    Returns
    -------
    a : (Mt,) updated amplitude
    history : list of sum-rate after each iter
    """
    Mt = a_init.size
    a = a_init.copy()
    a_prev = a.copy()
    history = []

    for t in range(1, iters + 1):
        # Current F = diag(a) Phi
        F = compute_F(a, Phi)

        # ----- 1) update mu_k (Eq. 51) -----
        HF = H @ F                                       # (Kc, N)
        sig = HF @ W                                     # (Kc, Kc + Ks)
        abs2 = np.abs(sig) ** 2
        mu = np.zeros(Kc)
        for k in range(Kc):
            num = abs2[k, k]
            denom = abs2[k, :].sum() - abs2[k, k] + sigma2
            mu[k] = num / max(denom, 1e-30)

        # ----- 2) update xi_k (Eq. 54) -----
        xi = np.zeros(Kc, dtype=complex)
        for k in range(Kc):
            denom = (abs2[k, :].sum() + sigma2)
            xi[k] = np.sqrt(1 + mu[k]) * (HF[k] @ W[:, k]) / max(denom, 1e-30)

        # ----- 3) build Q_bar, u_bar (Eq. 71-72) -----
        # D_k = diag(h_k); Q_k = D_k Phi W; total f1 quadratic form is
        #     sum_k |xi_k|^2 Q_k Q_k^H ;
        # in the manuscript, Q_bar = Re{ sum_k |xi_k|^2 Q_k Q_k^H }.
        Q_bar = np.zeros((Mt, Mt))
        u_bar = np.zeros(Mt)
        for k in range(Kc):
            Dk = H[k, :]                            # (Mt,)
            DkPhi = Dk[:, None] * Phi              # (Mt, N)
            Qk = DkPhi @ W                          # (Mt, Kc+Ks)
            # |xi_k|^2 * Q_k Q_k^H
            Q_bar += (np.abs(xi[k]) ** 2) * np.real(Qk @ Qk.conj().T)
            # 2 sqrt(1+mu_k) Re{ D_k Phi w_k xi_k^* }
            u_bar += 2.0 * np.sqrt(1 + mu[k]) * np.real((DkPhi @ W[:, k])
                                                        * np.conj(xi[k]))

        # ----- 4) accelerated PGD (Eq. 79-80) -----
        # lambda >= lambda_max(Q_bar). Use Frobenius norm as upper bound.
        lam = np.linalg.norm(Q_bar, ord='fro')
        lam = max(lam, 1e-9)

        if use_nesterov and t > 1:
            iota = max(0.0, (t - 2) / (t + 1))
            v = a + iota * (a - a_prev)
        else:
            v = a.copy()

        grad_term = u_bar - Q_bar @ v
        a_new = project_box(v + grad_term / lam, 0.0, 1.0)

        # bookkeeping
        a_prev = a
        a = a_new

        history.append(sum_rate(H, compute_F(a, Phi), W, sigma2, Kc))

    return a, history


def solve_SOOP1(scenario, sys_cfg, alg_cfg) -> Dict:
    """
    Run Algorithm 1 on a given scenario.

    Returns dict with W, A, history (list of sum-rates over outer iters).
    """
    rng = np.random.default_rng(sys_cfg.seed + 7)
    Mt, N = scenario.Mt, scenario.N
    Kc, Ks = scenario.Kc, scenario.Ks
    Pt = sys_cfg.Pt
    sigma2 = sys_cfg.sigma2

    a = rng.uniform(0.3, 1.0, size=Mt)
    Phi = scenario.Phi

    # Sensing precoder is zero in communication-centric design
    W_s = np.zeros((N, Ks), dtype=complex)

    history = {"sum_rate": [], "sensing_mi": []}
    # initial digital BF
    F = compute_F(a, Phi)
    H_eff = scenario.H @ F
    W_c, _ = _zf_waterfilling(H_eff, sigma2, Pt)
    W = np.concatenate([W_c, W_s], axis=1)

    for it in range(alg_cfg.outer_iters):
        # ---- (a) Digital BF: ZF + water-filling on current F ----
        F = compute_F(a, Phi)
        H_eff = scenario.H @ F
        W_c, _ = _zf_waterfilling(H_eff, sigma2, Pt)
        W = np.concatenate([W_c, W_s], axis=1)
        # safety scaling
        W = scale_W_to_power(F, W, Pt)

        # ---- (b) Holographic BF: SP2 ----
        a, _ = _sp2_inner(a, scenario.H, Phi, W, sigma2, Kc,
                          iters=alg_cfg.inner_iters,
                          use_nesterov=alg_cfg.nesterov)

        F = compute_F(a, Phi)
        # because a changes, re-scale W to fit power
        W = scale_W_to_power(F, W, Pt)

        history["sum_rate"].append(sum_rate(scenario.H, F, W, sigma2, Kc))
        from .utils import sensing_mi as _smi
        history["sensing_mi"].append(_smi(scenario.Bt_s, scenario.gamma_s2,
                                          F, W, sys_cfg.sigma_s2,
                                          sys_cfg.L, scenario.Mr))

        # convergence
        if it > 1 and abs(history["sum_rate"][-1]
                          - history["sum_rate"][-2]) < alg_cfg.tol:
            break

    return {
        "W": W, "A": np.diag(a), "a": a, "F": compute_F(a, Phi),
        "history": history,
    }
