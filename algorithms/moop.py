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
         - propose a by projected gradient ascent on the Lagrangian, then
           ACCEPT it via a monotonic backtracking line search on the original
           Tchebycheff objective tau (shrink the step until tau does not
           decrease). This is what guarantees the inner sequence is monotone;
           a plain PGD step does not.
         - update lambda_1 by projected gradient on g(lambda_1)
         - lambda_2 = (omega2 / I*) (1 - lambda_1 R*/omega1)

R*, I* are FIXED reference values throughout the MOOP iterations.
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
# SP5 fallback (no CVX): projected gradient ascent on the scalarised
# Tchebycheff objective tau = min(omega1*(R-R*)/R*, omega2*(I-I*)/I*).
# =====================================================================
def _sp5_fallback(scenario, sys_cfg, alg_cfg, a, W_prev, omega1, omega2,
                  R_star, I_star):
    """
    Approximately solve SP5 by projected gradient ascent on a smooth
    approximation of  min{ omega1*(R-R*)/R*,  omega2*(I-I*)/I* }.

    Uses log-sum-exp soft-min:
        tau_smooth = -beta^{-1} * log( exp(-beta*A1) + exp(-beta*A2) )
    where A1, A2 are the two normalised slack values.
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
    R_star = max(R_star, 1e-9)
    I_star = max(I_star, 1e-9)

    W = W_prev.copy()
    W = scale_W_to_power(F, W, Pt)

    # pre-compute b_t outer products needed for I gradient
    # I = sum_i log(1 + c_i * b_i^H F W W^H F^H b_i),  c_i = L Mr gamma_i^2/sigma_s2
    Bt = scenario.Bt_s                        # (Mt, Ks)
    c_sens = (L * Mr * scenario.gamma_s2) / sigma_s2

    beta_sm = 0.5   # smoother: both objectives contribute gradient
    best_tau = -np.inf
    best_W = W.copy()

    for it in range(40):
        FW = F @ W
        # ----- compute R, I and their gradients w.r.t. W -----
        # Communication: standard MU-MISO sum-rate gradient
        # SINR_k = |hk F wk|^2 / (sum_{j!=k}|hk F wj|^2 + ||hk F W^[s]||^2 + sigma2)
        # We keep all (Kc+Ks) columns active.
        HF = H @ F                            # (Kc, N)
        sig = HF @ W                          # (Kc, Kc+Ks)
        abs2 = np.abs(sig) ** 2

        grad_R = np.zeros_like(W)
        R_val = 0.0
        for k in range(Kc):
            inter = abs2[k, :].sum() - abs2[k, k] + sigma2
            sinr_k = abs2[k, k] / max(inter, 1e-30)
            R_val += np.log(1.0 + sinr_k)     # nats
            # dR_k/dW: standard MMSE-style gradient
            # numerator gradient
            ek = np.zeros(Kc + Ks); ek[k] = 1.0
            denom_total = abs2[k, :].sum() + sigma2
            # signal grad: 2 (HF[k])^H * sig[k,:] / denom_total  (but only col k)
            # interference grad: -sinr * 2 (HF[k])^H * sig[k,:] / denom_total (other cols)
            hk_F = HF[k:k+1]                   # (1, N)
            # grad of log(num/den + 1) = grad log(num+den) - grad log(den)
            num = abs2[k, k]
            den = inter
            num_plus_den = num + den
            # grad log(num+den) wrt W: 2 hk_F^H sig[k,:] / num_plus_den
            # grad log(den) wrt W:  same but with col k zeroed
            sig_k = sig[k:k+1, :]              # (1, Kc+Ks)
            sig_k_int = sig_k.copy()
            sig_k_int[0, k] = 0.0
            grad_R += (2.0 / max(num_plus_den, 1e-30)) * (hk_F.conj().T @ sig_k) \
                    - (2.0 / max(den, 1e-30)) * (hk_F.conj().T @ sig_k_int)

        # Sensing
        I_val = 0.0
        grad_I = np.zeros_like(W)
        FtF_per_b = []                         # for re-use
        for i in range(Ks):
            b = Bt[:, i]
            Fb = F.conj().T @ b                # (N,)
            FtF_per_b.append(Fb)
            gain = float(np.real(np.conj(Fb) @ (W @ W.conj().T) @ Fb))
            arg = c_sens[i] * gain
            I_val += np.log(1.0 + arg)
            # d/dW [Fb^H W W^H Fb] = 2 Fb Fb^H W
            grad_I += (2.0 * c_sens[i] / (1.0 + arg)) * np.outer(Fb, Fb.conj()) @ W

        # convert to bits/Hz so it matches R_star, I_star (both bits/Hz)
        ln2 = np.log(2.0)
        R_bits = R_val / ln2
        I_bits = I_val / ln2
        grad_R = grad_R / ln2
        grad_I = grad_I / ln2

        A1 = omega1 * (R_bits - R_star) / R_star
        A2 = omega2 * (I_bits - I_star) / I_star
        # soft-min weights (sum to 1, softmax of -beta*A)
        m = max(-A1, -A2)
        e1 = np.exp(-beta_sm * A1 + beta_sm * (-m))
        e2 = np.exp(-beta_sm * A2 + beta_sm * (-m))
        Z = e1 + e2 + 1e-30
        w1 = e1 / Z; w2 = e2 / Z
        tau_smooth = min(A1, A2)

        if tau_smooth > best_tau:
            best_tau = tau_smooth
            best_W = W.copy()

        # gradient of soft-min wrt W
        gW = w1 * (omega1 / R_star) * grad_R + w2 * (omega2 / I_star) * grad_I

        # normalized fixed step
        s = 0.01 / (np.linalg.norm(gW) + 1e-9)
        W = W + s * gW

        # project onto power constraint
        W = scale_W_to_power(F, W, Pt)

    return best_W, best_tau


# =====================================================================
# SP5 : digital BF (CVX)
# =====================================================================
def _solve_SP5(scenario, sys_cfg, alg_cfg, a, W_prev, R_star, I_star,
               omega1, omega2, outer_iter: int = 0):
    """
    Digital sub-problem with fixed amplitudes a (pseudocode lines 4-12).

    Adaptive SCA loop with two stopping criteria:
      1. n_sca(s) = max(1, round(sp5_iters / (1 + sp5_decay * s)))
         More steps in early outer iters, fewer in later ones.
      2. ΔW early-stop: if relative W change < sp5_tol, SCA has converged.
    Returns (W, tau).
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

    # if not _HAVE_CVX:
    #     return _sp5_fallback(scenario, sys_cfg, alg_cfg, a, W_prev,
    #                          omega1, omega2, R_star, I_star)

    R_star = max(R_star, 1e-9)
    I_star = max(I_star, 1e-9)

    # ── adaptive SCA budget: decreases harmonically with outer iteration ──
    # sp5_iters_max = getattr(alg_cfg, 'sp5_iters', 1)
    # sp5_decay     = getattr(alg_cfg, 'sp5_decay', 1.0)
    # sp5_tol       = getattr(alg_cfg, 'sp5_tol',   1e-3)
    sp5_tol=alg_cfg.sp5_tol
    # n_sca = max(1, round(sp5_iters_max / (1.0 + sp5_decay * outer_iter)))

    # g_vecs depend only on F (fixed in SP5), pre-compute once
    g_vecs = np.zeros((N, Ks), dtype=complex)
    for k in range(Ks):
        g_vecs[:, k] = np.sqrt(scenario.gamma_s2[k]) * (F.conj().T @ scenario.Bt_s[:, k])

    W_curr = scale_W_to_power(F, W_prev.copy(), Pt)
    tau_val = None
    success = False

    # ── SCA inner loop: for i in [1 : n_sca]  (pseudocode lines 5-11) ────
    for _ in range(alg_cfg.sp5_iters):
        # lines 6-8: update mu_k*, xi_k*, z_k = w_k^{(i-1)}
        sig  = HF @ W_curr
        abs2 = np.abs(sig) ** 2
        mu = np.zeros(Kc)
        xi = np.zeros(Kc, dtype=complex)
        for k in range(Kc):
            num    = abs2[k, k]
            denom  = abs2[k, :].sum() - abs2[k, k] + sigma2
            mu[k]  = num / max(denom, 1e-30)
            denom2 = abs2[k, :].sum() + sigma2
            xi[k]  = (np.sqrt(1 + mu[k]) * (HF[k] @ W_curr[:, k])
                      / max(denom2, 1e-30))

        Z_prev = W_curr.copy()                     # z_k^{(i)} = w_k^{(i-1)}

        # line 10: solve problem (60) — build CVX surrogate and solve
        W_var = cp.Variable((N, Kc + Ks), complex=True)
        tau   = cp.Variable()

        # f1_k: communication FP surrogate (Eq. 88)
        f1_terms = []
        for k in range(Kc):
            const_k = float(
                np.log(1 + mu[k]) - mu[k] - (np.abs(xi[k]) ** 2) * sigma2)
            hF_k = HF[k, :]
            lin  = 2.0 * np.sqrt(1 + mu[k]) * cp.real(
                cp.conj(xi[k]) * (hF_k @ W_var[:, k]))
            row  = hF_k @ W_var
            quad = (np.abs(xi[k]) ** 2) * cp.sum_squares(row)
            f1_terms.append(const_k + lin - quad)
        f1_sum = cp.sum(f1_terms)

        # g2: sensing FP surrogate (Eq. 90)
        g2_terms = []
        for k in range(Ks):
            gk      = g_vecs[:, k]
            lin_g   = 0
            const_g = 0.0
            for i in range(Kc + Ks):
                zi      = Z_prev[:, i]
                ai      = np.conj(zi) @ gk
                lin_g   = lin_g + 2.0 * cp.real(
                    np.conj(ai) * (gk.conj() @ W_var[:, i]))
                const_g += float(np.abs(ai) ** 2)
            # coef_k = (L * Mr * scenario.gamma_s2[k]) / sigma_s2
            coef_k = (L * Mr ) / sigma_s2
            g2_terms.append(cp.log1p(coef_k * (lin_g - const_g)))
        g2_sum = cp.sum(g2_terms)

        cons = [
            f1_sum >= (tau / omega1 + 1.0) * R_star * np.log(2.0),
            g2_sum >= (tau / omega2 + 1.0) * I_star * np.log(2.0),
            cp.sum_squares(F @ W_var) <= Pt,
        ]
        prob = cp.Problem(cp.Maximize(tau), cons)
        try:
            prob.solve(solver=alg_cfg.cvx_solver, verbose=alg_cfg.cvx_verbose)
        except Exception:
            prob.solve(solver="SCS", verbose=False)

        if W_var.value is None:
            break                                  # CVX failed; use best W so far

        W_new   = scale_W_to_power(F, np.asarray(W_var.value), Pt)
        # W_new   = np.asarray(W_var.value)
        tau_val = float(tau.value) if tau.value is not None else tau_val

        # ΔW early-stop: if W barely changed, further SCA steps won't help
        rel_dW = (np.linalg.norm(W_new - W_curr, 'fro')
                  / max(np.linalg.norm(W_curr, 'fro'), 1e-9))
        W_curr  = W_new
        success = True
        if rel_dW < sp5_tol:
            break                                  # SCA converged; stop early

    if not success:
        return _sp5_fallback(scenario, sys_cfg, alg_cfg, a, W_prev,
                             omega1, omega2, R_star, I_star)
    return W_curr, tau_val


# =====================================================================
# SP6 : amplitude BF via Lagrangian dual
# =====================================================================
def _solve_SP6(scenario, sys_cfg, alg_cfg, a_init, W, R_star, I_star,
               omega1, omega2,
               lam1_init: float = None, lam2_init: float = None,
               full_history: bool = False):
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
    inner_hist = {
        "R": [], "I": [], "duality_gap": [],
        "lam1": [], "lam2": [],
        "grad_norm": [], "da_norm": [],
    } if full_history else None

    # initial duals
    lam1_max = omega1 / R_star
    lam1 = lam1_init if lam1_init is not None else 0.5 * lam1_max
    lam2 = (omega2 / I_star) * (1.0 - lam1 * R_star / omega1)
    lam2 = max(lam2, 0.0)

    # best-iterate tracking: SP6 returns the highest-τ a seen, not the last
    def _tau_sp6(R, I):
        return min(omega1 * (R - R_star) / R_star,
                   omega2 * (I - I_star) / I_star)

    F0 = compute_F(a_init, Phi)
    R0_sp6 = sum_rate(scenario.H, F0, W, sigma2, Kc)
    I0_sp6 = sensing_mi(scenario.Bt_s, scenario.gamma_s2, F0, W,
                        sigma_s2, sys_cfg.L, Mr)
    best_tau_sp6  = _tau_sp6(R0_sp6, I0_sp6)
    best_a_sp6    = a_init.copy()
    best_lam1_sp6 = lam1
    best_lam2_sp6 = lam2

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
    # Each step does a projected-gradient ascent on the Lagrangian, but the
    # candidate amplitude is accepted ONLY if it does not decrease the
    # *original* Tchebycheff objective tau (monotonic backtracking line
    # search). A plain PGD step is not a strict MM/SCA update and can lower
    # the original objective even when the surrogate looks fine; the
    # backtracking makes the SP6 inner sequence monotone non-decreasing in tau.
    bt_beta      = getattr(alg_cfg, "bt_beta", 0.5)
    bt_max       = getattr(alg_cfg, "bt_max", 20)
    patience_max = getattr(alg_cfg, "MOOP_inner_patience", 15)
    no_progress  = 0

    for t in range(alg_cfg.MOOP_inner_iters):
        # ---- original objective at the current iterate (R*, I* are FIXED) ----
        F = compute_F(a, Phi)
        HF = H @ F
        R_cur = sum_rate(scenario.H, F, W, sigma2, Kc)
        I_cur = sensing_mi(scenario.Bt_s, scenario.gamma_s2, F, W,
                           sigma_s2, sys_cfg.L, Mr)
        tau_cur = _tau_sp6(R_cur, I_cur)

        # Update mu*, xi* (Lemma 1, Eq. 51, 54)
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
            ck = coeff_sens[k]
            Qk = ck * Q_sens[k]
            Qkz = Qk @ z
            zQz = float(np.real(z @ Qkz))
            zQa = float(np.real(z @ (Qk @ a)))
            denom = 1.0 - zQz + 2.0 * zQa
            grad_sens +=  2.0 * np.real(Qkz) / max(denom, 1e-12)
        grad_sens *= lam2

        grad_psi  = grad_comm + grad_sens
        grad_norm = float(np.linalg.norm(grad_psi))

        # ---- monotonic backtracking line search on the ORIGINAL tau ----
        #   a_cand = P_[0,1]( a + step * grad_psi );  accept iff
        #   tau(W, a_cand) >= tau(W, a),  else  step <- bt_beta * step.
        a_prev = a.copy()
        step = alg_cfg.pgd_step_a / (1.0 + grad_norm / max(Mt, 1))
        R_now, I_now, tau_now = R_cur, I_cur, tau_cur
        accepted = False
        for _bt in range(bt_max):
            a_cand = project_box(a_prev + step * grad_psi, 0.0, 1.0)
            F_cand = compute_F(a_cand, Phi)
            R_cand = sum_rate(scenario.H, F_cand, W, sigma2, Kc)
            I_cand = sensing_mi(scenario.Bt_s, scenario.gamma_s2, F_cand, W,
                                sigma_s2, sys_cfg.L, Mr)
            tau_cand = _tau_sp6(R_cand, I_cand)
            if tau_cand >= tau_cur - 1e-12:
                a = a_cand
                R_now, I_now, tau_now = R_cand, I_cand, tau_cand
                accepted = True
                break
            step *= bt_beta
        # if no step improves tau, keep a unchanged (stall); tau stays put
        da_norm = float(np.linalg.norm(a - a_prev))

        # ---- update lambda_1 via projected gradient on the dual function ----
        # (uses original R, I at the accepted amplitude)
        d_lam1 = (R_now - R_star) - (omega2 * R_star) / (omega1 * I_star) \
            * (I_now - I_star)
        lam1 = float(np.clip(lam1 - alg_cfg.pgd_step_lambda * d_lam1,
                             0.0, lam1_max))
        lam2 = max((omega2 / I_star) * (1.0 - lam1 * R_star / omega1), 0.0)

        # ---- convergence gap ----
        # The previous Lagrangian "duality gap" was ill-signed (it evaluated
        # tau_now - (tau_now + lam1) = -lam1 <= 0, so the stop test could fire
        # falsely) and was not a true dual function g(lambda)=sup_a L either.
        # With the monotone acceptance above, the honest, non-negative
        # convergence measure is the per-step improvement of the original
        # objective,  gap = tau^(t) - tau^(t-1) >= 0,  which -> 0 at a
        # stationary point.
        gap = max(tau_now - tau_cur, 0.0)

        # best-iterate safeguard (under monotone acceptance, best == latest)
        if tau_now > best_tau_sp6:
            best_tau_sp6  = tau_now
            best_a_sp6    = a.copy()
            best_lam1_sp6 = lam1
            best_lam2_sp6 = lam2

        if full_history:
            inner_hist["R"].append(R_now)
            inner_hist["I"].append(I_now)
            inner_hist["duality_gap"].append(gap)
            inner_hist["lam1"].append(lam1)
            inner_hist["lam2"].append(lam2)
            inner_hist["grad_norm"].append(grad_norm)
            inner_hist["da_norm"].append(da_norm)

        # ---- early stop: tau has stopped improving for `patience` steps ----
        if (not accepted) or gap < alg_cfg.tol:
            no_progress += 1
        else:
            no_progress = 0
        if no_progress >= patience_max:
            break

    # Return the best iterate seen (== latest under monotone acceptance)
    return best_a_sp6, best_lam1_sp6, best_lam2_sp6, inner_hist


# =====================================================================
# Top-level MOOP runner
# =====================================================================
def solve_MOOP(scenario, sys_cfg, alg_cfg,
               omega1: float = None, omega2: float = None,
               R_star: float = None, I_star: float = None,
               soop1_result=None, soop2_result=None,
               full_history: bool = False):
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
    a = rng.uniform(0.0, 1.0, size=Mt)

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
    if full_history:
        history["inner_R"] = []
        history["inner_I"] = []
        history["inner_tau"] = []
        history["inner_duality_gap"] = []
        # SP6-only per-inner-step detail (no SP5 prefix)
        history["sp6_lam1"] = []
        history["sp6_lam2"] = []
        history["sp6_grad_norm"] = []
        history["sp6_da_norm"] = []
    R_star_safe = max(R_star, 1e-9)
    I_star_safe = max(I_star, 1e-9)
    lam1_init   = None           # warm-started from SP6's best lam1 each outer iter

    def _tau(R, I):
        return min(omega1 * (R - R_star_safe) / R_star_safe,
                   omega2 * (I - I_star_safe) / I_star_safe)

    # initial tau
    R0 = sum_rate(scenario.H, F, W, sys_cfg.sigma2, Kc)
    I0 = sensing_mi(scenario.Bt_s, scenario.gamma_s2, F, W,
                    sys_cfg.sigma_s2, sys_cfg.L, scenario.Mr)
    best_tau = _tau(R0, I0)
    best_a, best_W = a.copy(), W.copy()
    best_R, best_I = R0, I0

    outer_tol     = getattr(alg_cfg, 'MOOP_outer_tol',     1e-4)
    outer_patience = getattr(alg_cfg, 'MOOP_outer_patience', 5)
    no_progress   = 0          # consecutive outer iters with Δτ < outer_tol

    for it in range(alg_cfg.MOOP_outer_iters):
        a_prev_iter, W_prev_iter = a.copy(), W.copy()

        # SP5: adaptive SCA loop; n_sca decreases with outer iter s=it
        W_new, tau_val = _solve_SP5(scenario, sys_cfg, alg_cfg,
                                    a, W, R_star, I_star, omega1, omega2,
                                    outer_iter=it)
        F = compute_F(a, Phi)
        # W_new = scale_W_to_power(F, W_new, Pt)

        # SP5 checkpoint for fine-grained history (before SP6 moves a)
        if full_history:
            R_sp5 = sum_rate(scenario.H, F, W_new, sys_cfg.sigma2, Kc)
            I_sp5 = sensing_mi(scenario.Bt_s, scenario.gamma_s2, F, W_new,
                               sys_cfg.sigma_s2, sys_cfg.L, scenario.Mr)

        # SP6: inner loop (for t in [1:T], pseudocode lines 17-30)
        # lam1_init warm-starts from SP6's best lam1 of the previous outer iter
        a_new, lam1_init, _, sp6_hist = _solve_SP6(
            scenario, sys_cfg, alg_cfg,
            a, W_new, R_star, I_star,
            omega1, omega2, lam1_init=lam1_init,
            full_history=full_history)
        F_new = compute_F(a_new, Phi)

        if full_history and sp6_hist is not None:
            outer_R   = [R_sp5] + sp6_hist["R"]
            outer_I   = [I_sp5] + sp6_hist["I"]
            outer_gap = [float("nan")] + sp6_hist["duality_gap"]
            history["inner_R"].append(outer_R)
            history["inner_I"].append(outer_I)
            history["inner_tau"].append([_tau(r, i) for r, i in zip(outer_R, outer_I)])
            history["inner_duality_gap"].append(outer_gap)
            history["sp6_lam1"].append(sp6_hist["lam1"])
            history["sp6_lam2"].append(sp6_hist["lam2"])
            history["sp6_grad_norm"].append(sp6_hist["grad_norm"])
            history["sp6_da_norm"].append(sp6_hist["da_norm"])
        W_new = scale_W_to_power(F_new, W_new, Pt)

        R_new = sum_rate(scenario.H, F_new, W_new, sys_cfg.sigma2, Kc)
        I_new = sensing_mi(scenario.Bt_s, scenario.gamma_s2, F_new, W_new,
                           sys_cfg.sigma_s2, sys_cfg.L, scenario.Mr)
        tau_new = _tau(R_new, I_new)

        da = np.linalg.norm(a_new - a_prev_iter)
        dW = np.linalg.norm(W_new - W_prev_iter, 'fro')

        # first iteration always accepted to give tau a free exploration step
        if it == 0 or tau_new >= best_tau - 1e-9:
            a, W = a_new, W_new
            best_tau = tau_new
            best_a, best_W = a.copy(), W.copy()
            best_R, best_I = R_new, I_new
            R_now, I_now = R_new, I_new
        else:
            # reject: keep previous iterate
            a, W = a_prev_iter, W_prev_iter
            F_now = compute_F(a, Phi)
            R_now = sum_rate(scenario.H, F_now, W, sys_cfg.sigma2, Kc)
            I_now = sensing_mi(scenario.Bt_s, scenario.gamma_s2, F_now, W,
                               sys_cfg.sigma_s2, sys_cfg.L, scenario.Mr)

        # Record tau of the RETAINED point so the coarse history stays
        # consistent with sum_rate/sensing_mi (and monotone non-decreasing).
        # Appending the candidate tau_new on a reject step is what made the
        # accepted-outer tau curve appear to drop.
        tau_now = _tau(R_now, I_now)
        print(f"  [MOOP it={it}] R={R_now:.3f} I={I_now:.3f} tau={tau_now:.4f}"
              f"  ||da||={da:.4f} ||dW||_F={dW:.4f}")

        history["sum_rate"].append(R_now)
        history["sensing_mi"].append(I_now)
        history["tau"].append(tau_now)

        # ── outer-loop early stopping: Δτ < outer_tol for outer_patience iters ──
        if it > 0:
            delta_tau = history["tau"][-1] - history["tau"][-2]
            if delta_tau < outer_tol:
                no_progress += 1
            else:
                no_progress = 0
            if no_progress >= outer_patience:
                print(f"  [MOOP] outer stopped: Δτ < {outer_tol:.0e} "
                      f"for {outer_patience} consecutive iters")
                break

    # always return the best iterate seen
    a, W = best_a, best_W
    F = compute_F(a, Phi)

    R_final = sum_rate(scenario.H, F, W, sys_cfg.sigma2, Kc)
    I_final = sensing_mi(scenario.Bt_s, scenario.gamma_s2, F, W,
                         sys_cfg.sigma_s2, sys_cfg.L, scenario.Mr)
    return {
        "W": W, "A": np.diag(a), "a": a, "F": F,
        "history": history,
        "R_star": R_star, "I_star": I_star,
        "omega1": omega1, "omega2": omega2,
        "tau_final": _tau(R_final, I_final),
        "relative_R": R_final / R_star_safe,
        "relative_I": I_final / I_star_safe,
    }
