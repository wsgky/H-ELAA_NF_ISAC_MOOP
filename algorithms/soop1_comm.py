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
import time
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
               iters: int = 10, pgd_steps: int = 20,
               ) -> Tuple[np.ndarray, list]:
    """
    Two-level inner loop for SP2 (amplitude update) — Algorithm 1.

    Outer loop (t = 0..iters-1): refreshes mu*, xi*, Q_bar, u_bar at best_a.
    Inner loop (s = 0..pgd_steps-1): Nesterov-accelerated PGD on the FIXED
    surrogate g = u^T a - a^T Q a.  Momentum is reset each outer step.

    Total PGD evaluations = iters * pgd_steps.
    R is recorded after every inner PGD step  →  len(history) = iters * pgd_steps.

    MM lower-bound chain:
        g(a; a_0) <= R(a)  (tight at a_0)
        ascending g  ⟹  ascending R
    Lipschitz constant of ∇g = u - 2*Q*a:  L = 2 * lambda_max(Q_bar).
    Correct PGD step:  alpha = 1 / (2 * lambda_max).
    """
    Mt = a_init.size
    a = a_init.copy()
    history = []

    best_R = sum_rate(H, compute_F(a, Phi), W, sigma2, Kc)
    best_a = a.copy()
    v_eig  = None

    for t in range(iters):
        # Build surrogate at best-known iterate for the strongest MM guarantee
        a = best_a.copy()

        # ── Update mu*, xi* once per surrogate refresh ───────────────────────
        F_t  = compute_F(a, Phi)
        HF   = H @ F_t                                    # (Kc, N)
        sig  = HF @ W                                     # (Kc, Kc+Ks)
        abs2 = np.abs(sig) ** 2
        mu = np.zeros(Kc)
        xi = np.zeros(Kc, dtype=complex)
        for k in range(Kc):
            num        = abs2[k, k]
            denom_int  = abs2[k, :].sum() - abs2[k, k] + sigma2
            mu[k]      = num / max(denom_int, 1e-30)
            denom_full = abs2[k, :].sum() + sigma2
            xi[k]      = (np.sqrt(1 + mu[k]) * (HF[k] @ W[:, k])
                          / max(denom_full, 1e-30))

        # ── Build Q_bar, u_bar (Eq. 71-72) ───────────────────────────────────
        Q_bar = np.zeros((Mt, Mt))
        u_bar = np.zeros(Mt)
        for k in range(Kc):
            Dk    = H[k, :]                              # (Mt,)
            DkPhi = Dk[:, None] * Phi                    # (Mt, N)
            Qk    = DkPhi @ W                            # (Mt, Kc+Ks)
            Q_bar += (np.abs(xi[k]) ** 2) * np.real(Qk @ Qk.conj().T)
            u_bar += (2.0 * np.sqrt(1 + mu[k])
                      * np.real((DkPhi @ W[:, k]) * np.conj(xi[k])))

        # ── lambda_max via warm-started power iteration ───────────────────────
        pi_iters = 40 if t == 0 else 15
        lam_raw, v_eig = _power_iter_lambda_max(Q_bar, iters=pi_iters, v0=v_eig)
        lam  = lam_raw * 1.05 + 1e-9
        step = 1.0 / (2.0 * lam)   # L = 2*lambda_max(Q_bar)

        # ── Nesterov-PGD on fixed surrogate (FISTA) ──────────────────────────
        a_pgd  = a.copy()
        a_prev = a.copy()
        t_nest = 1.0
        for s in range(pgd_steps):
            t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_nest ** 2)) / 2.0
            iota   = (t_nest - 1.0) / t_next
            v      = project_box(a_pgd + iota * (a_pgd - a_prev), 0.0, 1.0)

            a_new  = project_box(v + step * (u_bar - 2.0 * (Q_bar @ v)), 0.0, 1.0)

            a_prev = a_pgd
            a_pgd  = a_new
            t_nest = t_next

            R_now = sum_rate(H, compute_F(a_pgd, Phi), W, sigma2, Kc)
            history.append(R_now)
            if R_now > best_R + 1e-9:
                best_R = R_now
                best_a = a_pgd.copy()

    return best_a, history

def _sp2_inner_v2(a_init: np.ndarray, H: np.ndarray, Phi: np.ndarray,
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
            u_bar +=  np.sqrt(1 + mu[k]) * np.real((DkPhi @ W[:, k])
                                                        * np.conj(xi[k]))

        # ----- 4) accelerated PGD (Eq. 79-80) -----
        # lambda >= lambda_max(Q_bar). Use Frobenius norm as upper bound.
        lam = np.linalg.norm(Q_bar, ord='fro')
        lam = max(lam, 1e-9)
        # print(lam)
        # lam =5e4
        if use_nesterov and t > 1:
            iota = max(0.0, (t - 2) / (t + 1))
            v = a + iota * (a - a_prev)
        else:
            v = a.copy()

        grad_term = u_bar - Q_bar @ v
        # a_new = v + grad_term / lam
        a_new =project_box(v + grad_term / lam, 0.0, 1.0)
        # a_new =project_box(a + grad_term / lam, 0.0, 1.0)
        # bookkeeping
        a_prev = a
        a = a_new

        history.append(sum_rate(H, compute_F(a, Phi), W, sigma2, Kc))

    return a, history

def _power_iter_lambda_max(M: np.ndarray, iters: int = 25,
                           v0: np.ndarray = None
                           ) -> Tuple[float, np.ndarray]:
    """
    Estimate lambda_max of a symmetric PSD matrix via power iteration.

    Returns (lambda_max_estimate, final_eigenvector).
    Pass v0 to warm-start from a previous call's eigenvector.
    """
    n = M.shape[0]
    if n == 0:
        return 0.0, np.ones(1)
    if v0 is None:
        v = np.random.default_rng(0).standard_normal(n)
    else:
        v = v0.copy()
    v /= np.linalg.norm(v) + 1e-12
    lam = 0.0
    for _ in range(iters):
        v = M @ v
        nv = np.linalg.norm(v) + 1e-12
        lam = nv
        v = v / nv
    return float(lam), v


def solve_SOOP1(scenario, sys_cfg, alg_cfg,
                full_history: bool = False) -> Dict:
    """
    Run Algorithm 1 on a given scenario.

    Parameters
    ----------
    full_history : bool
        When True, disables early stopping and additionally stores
        ``history["inner_sum_rate"]``: a list of length outer_iters,
        where each element is ``[R_after_WF, R_inner_1, ..., R_inner_T]``
        — the R right after the ZF+WF step, then R after each inner PGD
        step.  Unroll across outer iters to get the fine-grained curve.

    Returns
    -------
    dict with W, A, a, F, history.
    history always contains "sum_rate" and "sensing_mi" (per outer iter).
    With full_history=True it also contains "inner_sum_rate".
    """
    from .utils import sensing_mi as _smi
    rng = np.random.default_rng(sys_cfg.seed + 7)
    Mt, N = scenario.Mt, scenario.N
    Kc, Ks = scenario.Kc, scenario.Ks
    Pt     = sys_cfg.Pt
    sigma2 = sys_cfg.sigma2
    Phi    = scenario.Phi
    W_s    = np.zeros((N, Ks), dtype=complex)

    a = rng.uniform(0, 1.0, size=Mt)

    history = {"sum_rate": [], "sensing_mi": []}
    if full_history:
        history["inner_sum_rate"] = []

    # initial digital BF
    F    = compute_F(a, Phi)
    W_c, _ = _zf_waterfilling(scenario.H, F, sigma2, Pt)
    W    = np.concatenate([W_c, W_s], axis=1)
    R_best = sum_rate(scenario.H, F, W, sigma2, Kc)
    a_best, W_best = a.copy(), W.copy()
    print(f"[SOOP1 speed testing:before SP2 iterations]")
    for it in range(alg_cfg.SOOP1_outer_iters):
        # ---- SP1: ZF + water-filling on current a -------------------------
        F    = compute_F(a, Phi)
        W_c, _ = _zf_waterfilling(scenario.H, F, sigma2, Pt)
        W    = np.concatenate([W_c, W_s], axis=1)
        R_after_wf = sum_rate(scenario.H, F, W, sigma2, Kc)

        # ---- SP2: amplitude update ----------------------------------------
        surr_iters = max(1, alg_cfg.SOOP1_inner_iters // alg_cfg.pgd_steps)
        print(f"[SOOP1 speed testing:SP2 outer iteration {it+1}/{alg_cfg.SOOP1_outer_iters}]")
        time_start = time.time()  # dummy timer using RNG calls
        a_new, sp2_hist = _sp2_inner(a, scenario.H, Phi, W, sigma2, Kc,
                                     iters=surr_iters,
                                     pgd_steps=alg_cfg.pgd_steps,
                                     )
        time_end = time.time()
        print(f"[SOOP1 speed testing:SP2 outer iteration {it+1}/{alg_cfg.SOOP1_outer_iters}] finished in {surr_iters} surrogate updates with {alg_cfg.SOOP1_inner_iters} PGD steps, took {time_end - time_start:.2f} seconds")
        
        if full_history:
            history["inner_sum_rate"].append([R_after_wf] + sp2_hist)

        # BCD-correct: re-solve SP1 at new a to get the true objective value
        F_new   = compute_F(a_new, Phi)
        W_c_new, _ = _zf_waterfilling(scenario.H, F_new, sigma2, Pt)
        W_new   = np.concatenate([W_c_new, W_s], axis=1)
        R_new   = sum_rate(scenario.H, F_new, W_new, sigma2, Kc)

        R_curr  = sum_rate(scenario.H, F, W, sigma2, Kc)
        if R_new >= R_curr - 1e-9:
            a, W = a_new, W_new
        # else: keep current a, W (rare)

        F     = compute_F(a, Phi)
        R_now = sum_rate(scenario.H, F, W, sigma2, Kc)
        history["sum_rate"].append(R_now)
        history["sensing_mi"].append(
            _smi(scenario.Bt_s, scenario.gamma_s2,
                 F, W, sys_cfg.sigma_s2, sys_cfg.L, scenario.Mr))

        if R_now > R_best:
            R_best = R_now
            a_best, W_best = a.copy(), W.copy()

        if not full_history:
            if it > 1 and abs(history["sum_rate"][-1]
                              - history["sum_rate"][-2]) < 1e-6:
                break

    a, W = a_best, W_best
    F = compute_F(a, Phi)
    return {
        "W": W, "A": np.diag(a), "a": a, "F": F,
        "history": history,
    }
