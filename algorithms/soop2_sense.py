"""
Algorithm 2 (SOOP2): sensing-centric beamforming.

Solves
    max  I = sum_i log(1 + (L sigma_i^2 / sigma^2) * b_i^H F W W^H F^H b_i)
    s.t. Tr(F W W^H F^H) <= Pt,  0 <= a_m <= 1.

Pipeline (manuscript Sec. V-B):

  Outer iteration i:
    1) SP3' (digital, SDP): max sum_i log(1 + (L/sigma^2) g_i^H Omega g_i)
       with g_i = sqrt(sigma_i^2) F^H b_i and Omega >> 0,
       Tr(F Omega F^H) <= Pt.
       Solved via CVX, then W = U Lambda^{1/2} from EVD of Omega*.
    2) SP4 (amplitude): build Q_k = D_k^H Phi W W^H Phi^H D_k for each ST,
       use first-order convex inequality to get a concave surrogate
       g_tilde and run projected gradient ascent (Eq. 87-88).
"""
import numpy as np
try:
    import cvxpy as cp
    _HAVE_CVX = True
except Exception:
    cp = None
    _HAVE_CVX = False

from .utils import (
    compute_F, sum_rate, sensing_mi, scale_W_to_power, project_box,
    transmit_power,
)


def _sp3_sdp(F: np.ndarray, Bt_s: np.ndarray, gamma_s2: np.ndarray,
             sigma_s2: float, L: int, Pt: float, N: int,
             solver: str = "SCS", verbose: bool = False) -> np.ndarray:
    """
    Solve SP3' via CVX:

        max  sum_i log(1 + (L / sigma_s2) g_i^H Omega g_i)
        s.t. Tr(F Omega F^H) <= Pt,  Omega >= 0.
    """
    Ks = Bt_s.shape[1]
    g = np.zeros((N, Ks), dtype=complex)
    for i in range(Ks):
        g[:, i] = np.sqrt(gamma_s2[i]) * (F.conj().T @ Bt_s[:, i])

    if not _HAVE_CVX:
        # ---- fallback: gradient ascent on Hermitian PSD Omega ----
        # parametrise Omega = M M^H,  with M (N, r) and r = Ks (=> rank Ks).
        rng = np.random.default_rng(0)
        r = max(Ks, 1)
        M = (rng.standard_normal((N, r)) + 1j * rng.standard_normal((N, r))) / np.sqrt(2)
        FHF = F.conj().T @ F
        # scale to budget
        p_now = float(np.real(np.trace(FHF @ (M @ M.conj().T))))
        if p_now > 0:
            M *= np.sqrt(Pt / p_now)
        for _ in range(50):
            Om = M @ M.conj().T
            grad = np.zeros_like(M)
            for i in range(Ks):
                gi = g[:, i:i+1]
                num = float(np.real(gi.conj().T @ Om @ gi).item())
                snr_i = (L / sigma_s2) * num
                weight = (L / sigma_s2) / (1.0 + snr_i)
                grad += weight * (gi @ (gi.conj().T @ M))
            # power penalty
            mu_pen = 0.5
            grad -= mu_pen * (FHF @ M)
            M = M + 1e-3 * grad
            # project to power budget
            p_now = float(np.real(np.trace(FHF @ (M @ M.conj().T))))
            if p_now > Pt and p_now > 0:
                M *= np.sqrt(Pt / p_now)
        return M @ M.conj().T

    Omega = cp.Variable((N, N), hermitian=True)
    constraints = [Omega >> 0]
    # power: real(Tr(F^H F Omega))
    FHF = F.conj().T @ F
    constraints.append(cp.real(cp.trace(FHF @ Omega)) <= Pt)

    obj_terms = []
    for i in range(Ks):
        gi = g[:, i:i + 1]
        # quad_form result is real-valued
        q = cp.real(cp.quad_form(gi, Omega))
        obj_terms.append(cp.log1p((L / sigma_s2) * q))
    obj = cp.Maximize(cp.sum(obj_terms))

    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=solver, verbose=verbose)
    except Exception as e:
        # fallback
        prob.solve(solver="SCS", verbose=False)

    Omega_val = Omega.value
    if Omega_val is None:
        # last resort: identity scaled to budget
        Omega_val = np.eye(N) * (Pt / max(np.real(np.trace(FHF)), 1e-9)) / N
    return np.asarray(Omega_val)


def _omega_to_W(Omega: np.ndarray, target_dim: int) -> np.ndarray:
    """
    EVD-based factorisation: W = U Lambda^{1/2} (size N x target_dim).
    Pad with zero columns if Omega has rank < target_dim.
    """
    Omega = 0.5 * (Omega + Omega.conj().T)  # Hermitianise
    eigvals, eigvecs = np.linalg.eigh(Omega)
    eigvals = np.maximum(eigvals.real, 0.0)
    # sort descending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    W = eigvecs * np.sqrt(eigvals)[None, :]    # (N, N)
    if W.shape[1] >= target_dim:
        return W[:, :target_dim]
    # pad
    pad = np.zeros((W.shape[0], target_dim - W.shape[1]), dtype=complex)
    return np.concatenate([W, pad], axis=1)


def _sp4_inner(a_init: np.ndarray, Phi: np.ndarray, W: np.ndarray,
               Bt_s: np.ndarray, gamma_s2: np.ndarray,
               sigma_s2: float, L: int, Mr: int,
               iters: int = 30, step: float = 5e-3
               ) -> np.ndarray:
    """
    SP4 amplitude update via projected gradient ascent on the surrogate
    g_tilde (Eq. 85). Step size is rescaled by 1 / (1 + ||Q_k|| sums).
    """
    Mt = a_init.size
    a = a_init.copy()

    # Pre-compute  D_k = diag(b_t_k);  Q_k = D_k^H Phi W W^H Phi^H D_k
    Ks = Bt_s.shape[1]
    Q_list = []
    coeff = []
    PhiW = Phi @ W                                 # (Mt, Kc+Ks)
    PhiW_PhiW_H = PhiW @ PhiW.conj().T             # (Mt, Mt)

    for k in range(Ks):
        bk = Bt_s[:, k]                            # (Mt,)
        # D_k = diag(bk);  D_k^H Phi W W^H Phi^H D_k = diag(bk^*) M diag(bk)
        Q = np.conj(bk)[:, None] * PhiW_PhiW_H * bk[None, :]
        Q_list.append(Q)
        coeff.append((L * Mr * gamma_s2[k]) / sigma_s2)

    # Adaptive step
    Q_norms = np.array([np.linalg.norm(Qk, 'fro') for Qk in Q_list])
    base = max(np.sum(Q_norms), 1.0)
    eff_step = step / base

    for t in range(iters):
        z = a.copy()       # surrogate anchor (Eq. 86: z^(t) = a^(t-1))
        # gradient of  sum_k log(1 + c_k * (2 z^T Q_k a - z^T Q_k z)).
        grad = np.zeros(Mt)
        for k in range(Ks):
            Qk = Q_list[k]
            ck = coeff[k]
            # work with real parts since the bilinear form is real for our Q
            Qz = Qk @ z
            zQz = float(np.real(z @ Qz))
            zQa = float(np.real(z @ (Qk @ a)))
            denom = 1.0 + ck * (2.0 * zQa - zQz)
            denom = max(denom, 1e-12)
            grad += ck * 2.0 * np.real(Qz) / denom
        a = project_box(a + eff_step * grad, 0.0, 1.0)

    return a


def solve_SOOP2(scenario, sys_cfg, alg_cfg):
    """
    Run Algorithm 2 on a given scenario.
    """
    rng = np.random.default_rng(sys_cfg.seed + 13)
    Mt, N = scenario.Mt, scenario.N
    Kc, Ks = scenario.Kc, scenario.Ks
    Pt = sys_cfg.Pt
    sigma_s2 = sys_cfg.sigma_s2

    a = rng.uniform(0.3, 1.0, size=Mt)
    Phi = scenario.Phi
    target_dim = Kc + Ks
    history = {"sum_rate": [], "sensing_mi": []}

    for it in range(alg_cfg.outer_iters):
        F = compute_F(a, Phi)

        # SP3' via CVX
        Omega = _sp3_sdp(F, scenario.Bt_s, scenario.gamma_s2,
                         sigma_s2, sys_cfg.L, Pt, N,
                         solver=alg_cfg.cvx_solver,
                         verbose=alg_cfg.cvx_verbose)
        W = _omega_to_W(Omega, target_dim)
        W = scale_W_to_power(F, W, Pt)

        # SP4: amplitude
        a = _sp4_inner(a, Phi, W,
                       scenario.Bt_s, scenario.gamma_s2,
                       sigma_s2, sys_cfg.L, scenario.Mr,
                       iters=alg_cfg.inner_iters,
                       step=alg_cfg.pgd_step_a)

        F = compute_F(a, Phi)
        W = scale_W_to_power(F, W, Pt)

        history["sum_rate"].append(sum_rate(scenario.H, F, W,
                                            sys_cfg.sigma2, Kc))
        history["sensing_mi"].append(sensing_mi(scenario.Bt_s,
                                                scenario.gamma_s2,
                                                F, W, sigma_s2,
                                                sys_cfg.L, scenario.Mr))
        if it > 1 and abs(history["sensing_mi"][-1]
                          - history["sensing_mi"][-2]) < alg_cfg.tol:
            break

    return {
        "W": W, "A": np.diag(a), "a": a, "F": compute_F(a, Phi),
        "history": history,
    }
