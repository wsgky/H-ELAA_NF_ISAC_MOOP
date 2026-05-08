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


def _zf_waterfilling(H: np.ndarray, F: np.ndarray, sigma2: float, Pt: float
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zero-forcing precoder + water-filling that satisfies the manuscript's
    transmit-power constraint  Tr(F W W^H F^H) <= Pt  exactly.

    Pipeline (manuscript Sec. V-A1):
        1) Effective channel:  H_bar = H @ F   (Kc x N)
        2) ZF precoder:  W_bar = H_bar^H (H_bar H_bar^H)^{-1}   (N x Kc)
        3) For w_k = beta_k W_bar[:,k] (with |beta_k|^2 = p_k_norm),
           ||F w_k||^2 = p_k_norm * ||F W_bar[:,k]||^2 ,
           and SINR_k = p_k_norm / sigma2  (since H_bar W_bar = I)
                      = p_k / (sigma2 * c_k)        with c_k := ||F W_bar[:,k]||^2.
        4) So the water-filling problem is
              max sum_k log(1 + p_k / (sigma2 * c_k))
              s.t.  sum_k p_k <= Pt.
           Closed form:  p_k = max(0, mu - sigma2 * c_k).
        5) Recover w_k = sqrt(p_k / c_k) * W_bar[:,k].

    Returns
    -------
    W_c : (N, Kc) precoding matrix
    p   : (Kc,) per-stream radiated power, sum <= Pt
    """
    H_bar = H @ F                                 # (Kc, N)
    Kc, N = H_bar.shape
    # ZF
    HHH = H_bar @ H_bar.conj().T
    HHH_inv = np.linalg.pinv(HHH)
    W_bar = H_bar.conj().T @ HHH_inv              # (N, Kc), satisfies H_bar W_bar = I

    # Radiated power per ZF column
    FW_bar = F @ W_bar                            # (Mt, Kc)
    c = np.real(np.sum(np.conj(FW_bar) * FW_bar, axis=0))  # ||F W_bar[:,k]||^2
    c = np.maximum(c, 1e-12)

    # Water-filling on the radiated-power budget
    eff_noise = sigma2 * c
    lo, hi = 0.0, eff_noise.max() + Pt + 1.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        p = np.maximum(0.0, mid - eff_noise)
        if p.sum() > Pt:
            hi = mid
        else:
            lo = mid
    p = np.maximum(0.0, mid - eff_noise)

    # Recover precoder. Set w_k = sqrt(p_k / c_k) * W_bar[:,k].
    # Then ||F w_k||^2 = p_k (radiated) and SINR_k = p_k / (sigma2 * c_k).
    W_c = W_bar * (np.sqrt(p / c))[None, :]
    return W_c, p


def _sp2_inner(a_init: np.ndarray, H: np.ndarray, Phi: np.ndarray,
               W: np.ndarray, sigma2: float, Kc: int,
               iters: int = 30, use_nesterov: bool = True
               ) -> Tuple[np.ndarray, list]:
    """
    Inner loop for SP2 (amplitude update) using FP + accelerated
    non-homogeneous quadratic transform.

    The auxiliary variables (mu, xi) are updated **only at t=0** so that
    the surrogate is fixed within the inner loop and we get strict
    monotonic ascent of the surrogate (manuscript's accelerated quadratic
    transform). Updating (mu, xi) every step turns the surrogate into a
    different function each iteration and breaks monotonicity of the
    original sum-rate.

    Returns
    -------
    a : (Mt,) updated amplitude
    history : list of sum-rate after each iter
    """
    Mt = a_init.size
    a = a_init.copy()
    a_prev = a.copy()
    history = []

    # ---- Update auxiliary variables ONCE per outer iteration ----
    F = compute_F(a, Phi)
    HF = H @ F                                       # (Kc, N)
    sig = HF @ W                                     # (Kc, Kc + Ks)
    abs2 = np.abs(sig) ** 2
    mu = np.zeros(Kc)
    xi = np.zeros(Kc, dtype=complex)
    for k in range(Kc):
        num = abs2[k, k]
        denom_int = abs2[k, :].sum() - abs2[k, k] + sigma2
        mu[k] = num / max(denom_int, 1e-30)
        denom_full = abs2[k, :].sum() + sigma2
        xi[k] = np.sqrt(1 + mu[k]) * (HF[k] @ W[:, k]) / max(denom_full, 1e-30)

    # ---- Build Q_bar (PSD), u_bar (Eq. 71-72) ONCE ----
    Q_bar = np.zeros((Mt, Mt))
    u_bar = np.zeros(Mt)
    for k in range(Kc):
        Dk = H[k, :]                                 # (Mt,)
        DkPhi = Dk[:, None] * Phi                    # (Mt, N)
        Qk = DkPhi @ W                               # (Mt, Kc+Ks)
        Q_bar += (np.abs(xi[k]) ** 2) * np.real(Qk @ Qk.conj().T)
        u_bar += 2.0 * np.sqrt(1 + mu[k]) * np.real((DkPhi @ W[:, k])
                                                    * np.conj(xi[k]))

    # ---- Lipschitz constant: lambda >= lambda_max(Q_bar) ----
    # Frobenius is a (often loose) upper bound; we use an estimate via a
    # short power iteration to get a tighter (and still safe) value.
    lam = _power_iter_lambda_max(Q_bar) * 1.01 + 1e-9

    best_R = sum_rate(H, compute_F(a, Phi), W, sigma2, Kc)
    best_a = a.copy()

    for t in range(1, iters + 1):
        # ---- accelerated PGD (Eq. 79-80) ----
        if use_nesterov and t > 1:
            iota = max(0.0, (t - 2) / (t + 1))
            v = a + iota * (a - a_prev)
            v = project_box(v, 0.0, 1.0)             # keep momentum feasible
        else:
            v = a.copy()

        grad_term = u_bar - Q_bar @ v
        a_new = project_box(v + grad_term / lam, 0.0, 1.0)

        a_prev = a
        a = a_new

        R_now = sum_rate(H, compute_F(a, Phi), W, sigma2, Kc)
        history.append(R_now)
        if R_now > best_R + 1e-9:
            best_R = R_now
            best_a = a.copy()

    # safeguard: return the best a seen (monotone-by-construction)
    return best_a, history


def _power_iter_lambda_max(M: np.ndarray, iters: int = 25) -> float:
    """Estimate the largest eigenvalue of a symmetric PSD matrix."""
    n = M.shape[0]
    if n == 0:
        return 0.0
    rng = np.random.default_rng(0)
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v) + 1e-12
    lam = 0.0
    for _ in range(iters):
        v = M @ v
        nv = np.linalg.norm(v) + 1e-12
        lam = nv
        v = v / nv
    return float(lam)


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
    W_c, _ = _zf_waterfilling(scenario.H, F, sigma2, Pt)
    W = np.concatenate([W_c, W_s], axis=1)
    R_best = sum_rate(scenario.H, F, W, sigma2, Kc)
    a_best, W_best = a.copy(), W.copy()

    for it in range(alg_cfg.outer_iters):
        # baseline at the start of this outer iteration
        R_curr = sum_rate(scenario.H, compute_F(a, Phi), W, sigma2, Kc)

        # ---- (a) Digital BF: ZF + water-filling on current F ----
        F = compute_F(a, Phi)
        W_c, _ = _zf_waterfilling(scenario.H, F, sigma2, Pt)
        W = np.concatenate([W_c, W_s], axis=1)
        # ZF + water-filling is the global optimum of SP1 for given a.

        # ---- (b) Holographic BF: SP2 ----
        a_new, sp2_hist = _sp2_inner(a, scenario.H, Phi, W, sigma2, Kc,
                                     iters=alg_cfg.inner_iters,
                                     use_nesterov=alg_cfg.nesterov)
        if sp2_hist:
            print(f"  [SOOP1 it={it}] SP2 R: {sp2_hist[0]:.4f} -> {sp2_hist[-1]:.4f}"
                  f"  (delta={sp2_hist[-1]-sp2_hist[0]:+.4f})")

        # BCD-correct: re-solve SP1 at the new a, then compare vs start-of-iter
        F_new = compute_F(a_new, Phi)
        W_c_new, _ = _zf_waterfilling(scenario.H, F_new, sigma2, Pt)
        W_new = np.concatenate([W_c_new, W_s], axis=1)
        R_new = sum_rate(scenario.H, F_new, W_new, sigma2, Kc)

        if R_new >= R_curr - 1e-9:
            a, W = a_new, W_new
        # else: keep a and W from SP1 block (rare in well-tuned BCD)

        F = compute_F(a, Phi)
        R_now = sum_rate(scenario.H, F, W, sigma2, Kc)
        history["sum_rate"].append(R_now)
        from .utils import sensing_mi as _smi
        history["sensing_mi"].append(_smi(scenario.Bt_s, scenario.gamma_s2,
                                          F, W, sys_cfg.sigma_s2,
                                          sys_cfg.L, scenario.Mr))

        if R_now > R_best:
            R_best = R_now
            a_best, W_best = a.copy(), W.copy()

        # convergence (tighter tolerance to avoid premature stop)
        if it > 1 and abs(history["sum_rate"][-1]
                          - history["sum_rate"][-2]) < 1e-6:
            break

    # always return the best iterate seen
    a, W = a_best, W_best
    F = compute_F(a, Phi)
    return {
        "W": W, "A": np.diag(a), "a": a, "F": F,
        "history": history,
    }
