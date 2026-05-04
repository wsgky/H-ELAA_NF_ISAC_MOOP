"""
Algorithm 3 (MOOP): jointly maximise R and I via the weighted Tchebycheff
scalarisation.

   max  tau
   s.t. omega1 (R - R*) / R*  >= tau
        omega2 (I - I*) / I*  >= tau
        Tr(F W W^H F^H) <= Pt,   0 <= a_m <= 1.

R*, I* are obtained by SOOP1 / SOOP2 first.

Pipeline (manuscript Sec. V-C):

  Outer iteration i:
    1) For given a, update aux. variables (mu_k, xi_k, z_i),
       solve SP5 (digital) via CVX:
            max tau
            s.t. sum_k f1_k({w_i}) >= (tau/omega1 + 1) R*
                 g2({w_i} | {z_i}) >= (tau/omega2 + 1) I*
                 Tr(W W^H) <= Pt'  (Pt' = Pt scaled by ||F||)
       This is convex in {w_i}.
    2) For given W, update aux. variables and run a Lagrangian-dual
       loop (Eq. 95-105) over (a, lambda_1, lambda_2):
         - update a by projected gradient ascent on the Lagrangian
         - update lambda_1 by projected gradient on g(lambda_1)
         - lambda_2 = (omega2 / I*) (1 - lambda_1 R*/omega1)
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
from .soop1_comm import solve_SOOP1
from .soop2_sense import solve_SOOP2


# =====================================================================
# SP5 fallback (no CVX): gradient ascent on a weighted (R + I) objective
# =====================================================================
def _sp5_fallback(scenario, sys_cfg, alg_cfg, a, W_prev, omega1, omega2):
    """
    When CVX is unavailable, take a short PGD on W to approximately
    improve the weighted objective. This won't give exact tau but is
    sufficient for end-to-end smoke testing.
    """
    Phi = scenario.Phi
    H = scenario.H
    Kc, Ks = scenario.Kc, scenario.Ks
    Mr = scenario.Mr
    N = scenario.N
    Pt = sys_cfg.Pt
    sigma2 = sys_cfg.sigma2
    sigma_s2 = sys_cfg.sigma_s2
    L = sys_cfg.L
    F = compute_F(a, Phi)

    W = W_prev.copy()

    for _ in range(8):
        # finite-difference style gradient on R+I via analytic forms
        FW = F @ W
        # gradient of sum_k log(1+SINR_k) w.r.t. W (use a simple zero-grad
        # approximation: drive W toward MRT of effective channel)
        # MRT step
        H_eff = H @ F                          # (Kc, N)
        # normalise rows
        for k in range(Kc):
            denom = np.linalg.norm(H_eff[k]) + 1e-12
            W[:, k] = 0.7 * W[:, k] + 0.3 * (H_eff[k].conj() / denom) \
                * np.sqrt(Pt / (Kc + Ks))

        # sensing streams: align with strongest ST
        for s in range(Ks):
            g = F.conj().T @ scenario.Bt_s[:, s]
            denom = np.linalg.norm(g) + 1e-12
            W[:, Kc + s] = 0.7 * W[:, Kc + s] + 0.3 * (g / denom) \
                * np.sqrt(omega2 * Pt / (Kc + Ks))

        # power scaling
        W = scale_W_to_power(F, W, Pt)

    # estimate tau roughly
    R_now = sum_rate(scenario.H, F, W, sigma2, Kc)
    I_now = sensing_mi(scenario.Bt_s, scenario.gamma_s2, F, W,
                       sigma_s2, L, Mr)
    return W, None


# =====================================================================
# SP5 : digital BF (CVX)
# =====================================================================
def _solve_SP5(scenario, sys_cfg, alg_cfg, a, W_prev, R_star, I_star,
               omega1, omega2):
    """
    Digital sub-problem with fixed amplitudes a. Returns updated W.
    """
    Phi = scenario.Phi
    H = scenario.H
    Kc, Ks = scenario.Kc, scenario.Ks
    N, Mt = scenario.N, scenario.Mt
    Pt = sys_cfg.Pt
    sigma2 = sys_cfg.sigma2
    sigma_s2 = sys_cfg.sigma_s2
    L = sys_cfg.L
    Mr = scenario.Mr

    F = compute_F(a, Phi)
    HF = H @ F                                     # (Kc, N)

    if not _HAVE_CVX:
        # ---- fallback: do a few projected-gradient steps on W ----
        # Maximise w.r.t. W:  J = R(W) + I(W)  weighted by current
        # coefficients (omega1/omega2) -- a heuristic but feasible warm-start.
        return _sp5_fallback(scenario, sys_cfg, alg_cfg, a, W_prev,
                             omega1, omega2)

    # ---- compute aux variables mu*, xi* with the previous W ----
    sig = HF @ W_prev                              # (Kc, Kc+Ks)
    abs2 = np.abs(sig) ** 2
    mu = np.zeros(Kc)
    xi = np.zeros(Kc, dtype=complex)
    for k in range(Kc):
        num = abs2[k, k]
        denom = abs2[k, :].sum() - abs2[k, k] + sigma2
        mu[k] = num / max(denom, 1e-30)
        denom2 = abs2[k, :].sum() + sigma2
        xi[k] = np.sqrt(1 + mu[k]) * (HF[k] @ W_prev[:, k]) / max(denom2, 1e-30)

    # ---- z_i = w_i^{(prev)} (Eq. 89, choice z = a) ----
    Z_prev = W_prev.copy()                         # (N, Kc+Ks)

    # ---- coefficients for f1_k(w) (linear+quadratic in W) ----
    # f1_k = log(1+mu_k) - mu_k - |xi_k|^2 sigma_k^2
    #        + 2 sqrt(1+mu_k) Re{ h_k F w_k xi_k^* }
    #        - |xi_k|^2 || h_k F W ||^2
    # The first two terms are constants for the optimisation.

    # ---- coefficients for g2 (Eq. 90) ----
    # G_k = g_k g_k^H,  g_k = sqrt(gamma_k^2) F^H b_k
    g_vecs = np.zeros((N, Ks), dtype=complex)
    for k in range(Ks):
        g_vecs[:, k] = np.sqrt(scenario.gamma_s2[k]) * (F.conj().T @ scenario.Bt_s[:, k])

    # ---- CVX variables ----
    # Stack W as (N, Kc+Ks) complex variable
    W_var = cp.Variable((N, Kc + Ks), complex=True)
    tau = cp.Variable()

    # f1_k builder
    f1_terms = []
    for k in range(Kc):
        const_k = float(np.log(1 + mu[k]) - mu[k] - (np.abs(xi[k]) ** 2) * sigma2)
        # 2 sqrt(1+mu_k) Re{ (h_k F) w_k xi_k^* }
        hF_k = HF[k, :]                           # (N,)
        lin = 2.0 * np.sqrt(1 + mu[k]) * cp.real(
            cp.conj(xi[k]) * (hF_k @ W_var[:, k])
        )
        # |xi_k|^2 || h_k F W ||^2  =  |xi_k|^2 * sum_j |hF_k @ w_j|^2
        # (hF_k @ W_var) is a 1-by-(Kc+Ks) row vector
        row = hF_k @ W_var                       # row vector
        quad = (np.abs(xi[k]) ** 2) * cp.sum_squares(row)
        f1_terms.append(const_k + lin - quad)

    f1_sum = cp.sum(f1_terms)

    # g2 builder
    # log(1 + sum_i 2 Re{z_i^H G_k w_i} - z_i^H G_k z_i),  k = 1..Ks.
    g2_terms = []
    for k in range(Ks):
        gk = g_vecs[:, k]                         # (N,)
        lin_g = 0
        const_g = 0
        for i in range(Kc + Ks):
            zi = Z_prev[:, i]
            # z_i^H g_k g_k^H w_i = (z_i^H g_k)*(g_k^H w_i)  -- scalar
            ai = np.conj(zi) @ gk                 # scalar
            # 2 Re{ ai^* * (g_k^H w_i) }
            lin_g = lin_g + 2.0 * cp.real(np.conj(ai) * (gk.conj() @ W_var[:, i]))
            const_g += float(np.abs(ai) ** 2)
        coef_k = (L * Mr * scenario.gamma_s2[k]) / sigma_s2
        g2_terms.append(cp.log1p(coef_k * (lin_g - const_g)))
    g2_sum = cp.sum(g2_terms)

    # ---- power constraint: Tr(F W W^H F^H) = || F W ||_F^2 ----
    FW = F @ W_var
    pow_con = cp.sum_squares(FW) <= Pt

    # ---- the two scalarisation constraints ----
    R_star = max(R_star, 1e-9)
    I_star = max(I_star, 1e-9)
    cons = [
        f1_sum >= (tau / omega1 + 1.0) * R_star * np.log(2.0),  # convert log2->ln
        g2_sum >= (tau / omega2 + 1.0) * I_star * np.log(2.0),
        pow_con,
    ]

    prob = cp.Problem(cp.Maximize(tau), cons)
    try:
        prob.solve(solver=alg_cfg.cvx_solver, verbose=alg_cfg.cvx_verbose)
    except Exception:
        prob.solve(solver="SCS", verbose=False)

    if W_var.value is None:
        return W_prev, None
    W_new = np.asarray(W_var.value)
    # safety
    W_new = scale_W_to_power(F, W_new, Pt)
    return W_new, float(tau.value) if tau.value is not None else None


# =====================================================================
# SP6 : amplitude BF via Lagrangian dual
# =====================================================================
def _solve_SP6(scenario, sys_cfg, alg_cfg, a_init, W, R_star, I_star,
               omega1, omega2,
               lam1_init: float = None, lam2_init: float = None):
    """
    Amplitude sub-problem with fixed W. Lagrangian-dual ascent on
    (a, lambda_1, lambda_2).
    """
    Phi = scenario.Phi
    H = scenario.H
    Kc, Ks = scenario.Kc, scenario.Ks
    Mt = scenario.Mt
    sigma2 = sys_cfg.sigma2
    sigma_s2 = sys_cfg.sigma_s2
    L = sys_cfg.L
    Mr = scenario.Mr

    a = a_init.copy()
    R_star = max(R_star, 1e-9)
    I_star = max(I_star, 1e-9)

    # initial duals
    lam1_max = omega1 / R_star
    lam1 = lam1_init if lam1_init is not None else 0.5 * lam1_max
    lam2 = (omega2 / I_star) * (1.0 - lam1 * R_star / omega1)
    lam2 = max(lam2, 0.0)

    # ---- pre-compute pieces that don't depend on a ----
    # Q_bar(W), u_bar(W) for the comm surrogate (Eq. 71-72)
    PhiW = Phi @ W                                 # (Mt, Kc+Ks)

    # For sensing: Q_k = D_k^H Phi W W^H Phi^H D_k  with D_k = diag(b_k)
    PhiW_PhiW_H = PhiW @ PhiW.conj().T             # (Mt, Mt)
    Q_sens = []
    coeff_sens = []
    for k in range(Ks):
        bk = scenario.Bt_s[:, k]
        Qk = np.conj(bk)[:, None] * PhiW_PhiW_H * bk[None, :]
        Q_sens.append(Qk)
        coeff_sens.append((L * Mr * scenario.gamma_s2[k]) / sigma_s2)

    # ---- inner loop on a, lam1 ----
    for t in range(alg_cfg.inner_iters):
        # Update mu*, xi* (Lemma 1, Eq. 51, 54)
        F = compute_F(a, Phi)
        HF = H @ F
        sig = HF @ W
        abs2 = np.abs(sig) ** 2
        mu = np.zeros(Kc)
        xi = np.zeros(Kc, dtype=complex)
        for k in range(Kc):
            num = abs2[k, k]
            denom = abs2[k, :].sum() - abs2[k, k] + sigma2
            mu[k] = num / max(denom, 1e-30)
            denom2 = abs2[k, :].sum() + sigma2
            xi[k] = np.sqrt(1 + mu[k]) * (HF[k] @ W[:, k]) / max(denom2, 1e-30)

        # Build Q_bar (real PSD) and u_bar (Eq. 71, 72) for the comm surrogate
        Q_bar = np.zeros((Mt, Mt))
        u_bar = np.zeros(Mt)
        for k in range(Kc):
            Dk = H[k, :]
            DkPhi = Dk[:, None] * Phi
            Qk = DkPhi @ W
            Q_bar += (np.abs(xi[k]) ** 2) * np.real(Qk @ Qk.conj().T)
            u_bar += 2.0 * np.sqrt(1 + mu[k]) * np.real(
                (DkPhi @ W[:, k]) * np.conj(xi[k]))

        # ---- gradient of Lagrangian wrt a ----
        # comm part:  lambda_1 * ( -2 Q_bar a + u_bar )
        # NOTE: this is the gradient of  sum_k f1_k_quadratic = -a^T Q a + a^T u + const
        grad_comm = lam1 * (-2.0 * (Q_bar @ a) + u_bar)

        # sensing part: lambda_2 * sum_k 2 Q_k z / (c_k + z^T Q_k a)
        # using z = a (Eq. 86)
        z = a.copy()
        grad_sens = np.zeros(Mt)
        for k in range(Ks):
            Qk = Q_sens[k]
            ck = coeff_sens[k]
            Qkz = Qk @ z
            zQz = float(np.real(z @ Qkz))
            zQa = float(np.real(z @ (Qk @ a)))
            denom = 1.0 + ck * (2.0 * zQa - zQz)
            grad_sens += ck * 2.0 * np.real(Qkz) / max(denom, 1e-12)
        grad_sens *= lam2

        grad_psi = grad_comm + grad_sens

        # adaptive step (rescale)
        step_a = alg_cfg.pgd_step_a / (1.0 + np.linalg.norm(grad_psi) / max(Mt, 1))
        a = project_box(a + step_a * grad_psi, 0.0, 1.0)

        # ---- update lambda_1 via projected gradient on the dual function ----
        F = compute_F(a, Phi)
        # current f1_sum / log2 sum_rate
        # f1_sum is in nats; scale R* by ln(2) to compare with our log2 R_star
        R_now = sum_rate(scenario.H, F, W, sigma2, Kc)
        I_now = sensing_mi(scenario.Bt_s, scenario.gamma_s2, F, W,
                           sigma_s2, sys_cfg.L, Mr)
        # dg/dlam1 = (R_now - R*) - (omega2 R*) / (omega1 I*) * (I_now - I*)
        d_lam1 = (R_now - R_star) - (omega2 * R_star) / (omega1 * I_star) \
            * (I_now - I_star)
        lam1 = lam1 - alg_cfg.pgd_step_lambda * d_lam1
        lam1 = float(np.clip(lam1, 0.0, lam1_max))
        lam2 = (omega2 / I_star) * (1.0 - lam1 * R_star / omega1)
        lam2 = max(lam2, 0.0)

    return a, lam1, lam2


# =====================================================================
# Top-level MOOP runner
# =====================================================================
def solve_MOOP(scenario, sys_cfg, alg_cfg,
               omega1: float = None, omega2: float = None,
               R_star: float = None, I_star: float = None,
               soop1_result=None, soop2_result=None):
    """
    Run Algorithm 3 (MOOP). If R*, I* are not provided, run SOOP1/SOOP2
    first to get them.

    Returns dict with W, a, history including R, I and tau per iter.
    """
    if omega1 is None:
        omega1 = alg_cfg.omega1
    if omega2 is None:
        omega2 = alg_cfg.omega2

    # ---- step 0: get reference points if needed ----
    if R_star is None:
        if soop1_result is None:
            soop1_result = solve_SOOP1(scenario, sys_cfg, alg_cfg)
        R_star = soop1_result["history"]["sum_rate"][-1]
    if I_star is None:
        if soop2_result is None:
            soop2_result = solve_SOOP2(scenario, sys_cfg, alg_cfg)
        I_star = soop2_result["history"]["sensing_mi"][-1]

    rng = np.random.default_rng(sys_cfg.seed + 21)
    Phi = scenario.Phi
    Mt, N = scenario.Mt, scenario.N
    Kc, Ks = scenario.Kc, scenario.Ks
    Pt = sys_cfg.Pt

    # init: warm-start with a uniform amplitude and a feasible W
    a = rng.uniform(0.5, 1.0, size=Mt)

    F = compute_F(a, Phi)
    # Use SOOP1 + SOOP2 W as warm-start: average them after EVD-fix to N
    if soop1_result is not None and soop2_result is not None:
        W = 0.5 * soop1_result["W"] + 0.5 * soop2_result["W"]
    elif soop1_result is not None:
        W = soop1_result["W"]
    elif soop2_result is not None:
        W = soop2_result["W"]
    else:
        # random
        W = (rng.standard_normal((N, Kc + Ks))
             + 1j * rng.standard_normal((N, Kc + Ks))) / np.sqrt(2)
    W = scale_W_to_power(F, W, Pt)

    history = {"sum_rate": [], "sensing_mi": [], "tau": []}
    lam1_init = None
    for it in range(alg_cfg.outer_iters):
        # SP5
        W, tau_val = _solve_SP5(scenario, sys_cfg, alg_cfg,
                                a, W, R_star, I_star, omega1, omega2)
        F = compute_F(a, Phi)
        W = scale_W_to_power(F, W, Pt)
        # SP6
        a, lam1_init, _ = _solve_SP6(scenario, sys_cfg, alg_cfg,
                                     a, W, R_star, I_star, omega1, omega2,
                                     lam1_init=lam1_init)
        F = compute_F(a, Phi)
        W = scale_W_to_power(F, W, Pt)

        R_now = sum_rate(scenario.H, F, W, sys_cfg.sigma2, Kc)
        I_now = sensing_mi(scenario.Bt_s, scenario.gamma_s2, F, W,
                           sys_cfg.sigma_s2, sys_cfg.L, scenario.Mr)
        history["sum_rate"].append(R_now)
        history["sensing_mi"].append(I_now)
        history["tau"].append(tau_val)

        if it > 1:
            d1 = abs(history["sum_rate"][-1] - history["sum_rate"][-2])
            d2 = abs(history["sensing_mi"][-1] - history["sensing_mi"][-2])
            if d1 < alg_cfg.tol and d2 < alg_cfg.tol:
                break

    return {
        "W": W, "A": np.diag(a), "a": a, "F": compute_F(a, Phi),
        "history": history,
        "R_star": R_star, "I_star": I_star,
        "omega1": omega1, "omega2": omega2,
    }
