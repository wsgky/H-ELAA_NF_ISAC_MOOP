"""
Algorithm 3 (MOOP): jointly maximise R and I via the weighted Tchebycheff
scalarisation.

   max  tau
   s.t. omega1 (R - R*) / R*  >= tau
        omega2 (I - I*) / I*  >= tau
        Tr(F W W^H F^H) <= Pt,   0 <= a_m <= 1.

R*, I* are obtained by SOOP1 / SOOP2 first.

Algorithm 1 (manuscript Sec. V-C and pseudocode):

  Outer iteration s = 1,...,S:
    1) Digital-beamforming update (SP5, fixed A^(s-1), lines 4-13):
         max tau  s.t.
           R̲(W|μ*,ξ*)           >= (tau/omega1 + 1) R*       [Eq. 63b]
           I̲_W({w_i}|{z_i})     >= (tau/omega2 + 1) I*       [Eq. 63c]
           d_k^W({w_i}|{z_i})   >= eps_dom  for all k        [Eq. 63d]
           Tr(F(sum_i w_i w_i^H)F^H) <= Pt                  [Eq. 63e]
         mu*, xi* updated once per SCA step (lines 7, Eq. 49-50).
         Inner stopping: |tau_W^(i) - tau_W^(i-1)| <= eps    [line 9]

    2) Holographic-amplitude update (SP6, fixed W^(s), lines 14-29):
         lambda1^(s,0) ~ Unif[0, omega1/R*]                  [line 16]
         Inner loop (lines 18-27):
           (a) Update mu*, xi* (Eq. 49-50)                   [line 19]
           (b) z = a^(t-1)                                   [line 20]
           (c) Update lambda1, lambda2 FIRST (Eq. 76-78)     [line 21]
           (d) Compute grad of psi wrt a using new lambdas   [Eq. 79]
           (e) Update a via BTLS projected gradient (Eq. 80) [line 22]
           (f) Recover tau^(t) (Eq. 81)                      [line 23]
         Inner stopping: Δ_a <= eps AND Δ_in <= eps          [line 25]

  Outer stopping: ||ΔW|| <= eps AND ||Δa|| <= eps AND |Δtau| <= eps
                                                              [lines 31-33]

R*, I* are FIXED reference values throughout the MOOP iterations.
"""
import numpy as np
try:
    import cvxpy as cp
    _HAVE_CVX = True
    # _HAVE_CVX = False
except Exception:
    cp = None
    _HAVE_CVX = False

from .utils import (
    compute_F, sum_rate, sensing_mi, scale_W_to_power,
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

    Bt = scenario.Bt_s                        # (Mt, Ks)
    c_sens = (L * Mr * scenario.gamma_s2) / sigma_s2

    beta_sm = 0.5
    best_tau = -np.inf
    best_W = W.copy()

    for _ in range(40):
        FW = F @ W
        HF = H @ F                            # (Kc, N)
        sig = HF @ W                          # (Kc, Kc+Ks)
        abs2 = np.abs(sig) ** 2

        grad_R = np.zeros_like(W)
        R_val = 0.0
        for k in range(Kc):
            inter = abs2[k, :].sum() - abs2[k, k] + sigma2
            sinr_k = abs2[k, k] / max(inter, 1e-30)
            R_val += np.log(1.0 + sinr_k)
            ek = np.zeros(Kc + Ks); ek[k] = 1.0
            denom_total = abs2[k, :].sum() + sigma2
            hk_F = HF[k:k+1]                   # (1, N)
            num = abs2[k, k]
            den = inter
            num_plus_den = num + den
            sig_k = sig[k:k+1, :]              # (1, Kc+Ks)
            sig_k_int = sig_k.copy()
            sig_k_int[0, k] = 0.0
            grad_R += (2.0 / max(num_plus_den, 1e-30)) * (hk_F.conj().T @ sig_k) \
                    - (2.0 / max(den, 1e-30)) * (hk_F.conj().T @ sig_k_int)

        I_val = 0.0
        grad_I = np.zeros_like(W)
        FtF_per_b = []
        for i in range(Ks):
            b = Bt[:, i]
            Fb = F.conj().T @ b                # (N,)
            FtF_per_b.append(Fb)
            gain = float(np.real(np.conj(Fb) @ (W @ W.conj().T) @ Fb))
            arg = c_sens[i] * gain
            I_val += np.log(1.0 + arg)
            grad_I += (2.0 * c_sens[i] / (1.0 + arg)) * np.outer(Fb, Fb.conj()) @ W

        ln2 = np.log(2.0)
        R_bits = R_val / ln2
        I_bits = I_val / ln2
        grad_R = grad_R / ln2
        grad_I = grad_I / ln2

        A1 = omega1 * (R_bits - R_star) / R_star
        A2 = omega2 * (I_bits - I_star) / I_star
        m = max(-A1, -A2)
        e1 = np.exp(-beta_sm * A1 + beta_sm * (-m))
        e2 = np.exp(-beta_sm * A2 + beta_sm * (-m))
        Z = e1 + e2 + 1e-30
        w1 = e1 / Z; w2 = e2 / Z
        tau_smooth = min(A1, A2)

        if tau_smooth > best_tau:
            best_tau = tau_smooth
            best_W = W.copy()

        gW = w1 * (omega1 / R_star) * grad_R + w2 * (omega2 / I_star) * grad_I

        s = 0.01 / (np.linalg.norm(gW) + 1e-9)
        W = W + s * gW
        W = scale_W_to_power(F, W, Pt)

    return best_W, best_tau


# =====================================================================
# SP5 : digital BF (CVX)  — Algorithm 1 lines 4-13
# =====================================================================
def _solve_SP5(scenario, sys_cfg, alg_cfg, a, W_prev, R_star, I_star,
               omega1, omega2, outer_iter: int = 0, tau_init: float = None):
    """
    Digital sub-problem with fixed amplitudes a (Algorithm 1 lines 4-13).

    Solving Eq. (63) via CVX SCA:
      - mu*, xi* updated once per SCA step (line 7, Eq. 49-50).
      - Domain constraint d_k^W >= eps_dom added (Eq. 63d).
      - Stopping: |tau_W^(i) - tau_W^(i-1)| <= tol (line 9).

    tau_init: tau^(s-1) from the previous outer iteration, used as
              tau_W^(0) for the inner stopping check (line 5).
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

    R_star = max(R_star, 1e-9)
    I_star = max(I_star, 1e-9)

    sp5_tol = alg_cfg.sp5_tol
    eps_dom = getattr(alg_cfg, 'eps_dom', 1e-6)

    # g_vecs depend only on F (fixed in SP5), pre-compute once
    g_vecs = np.zeros((N, Ks), dtype=complex)
    for k in range(Ks):
        g_vecs[:, k] = np.sqrt(scenario.gamma_s2[k]) * (F.conj().T @ scenario.Bt_s[:, k])

    W_curr = scale_W_to_power(F, W_prev.copy(), Pt)
    tau_val = tau_init       # tau_W^(0) = tau^(s-1)  [Algorithm 1 line 5]
    success = False

    # SCA inner loop: lines 6-12 of Algorithm 1
    for _ in range(alg_cfg.sp5_iters):
        # lines 7: update mu_k*, xi_k* (Eq. 49-50)
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

        Z_prev = W_curr.copy()                     # z_k^(i) = w_k^(i-1)

        if not _HAVE_CVX:
            print("[**********Error**********]: SP5 is not solved via CVX]")
            break

        # line 8: solve Eq. (63)
        W_var = cp.Variable((N, Kc + Ks), complex=True)
        tau   = cp.Variable()

        # f1_k: communication FP surrogate (Eq. 55 / Lemma 1)
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

        # g2: sensing FP surrogate (Eq. 55), build d_k^W expressions
        g2_exprs = []
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
            coef_k = (L * Mr) / sigma_s2
            g2_exprs.append(coef_k * (lin_g - const_g))

        g2_sum = cp.sum([cp.log1p(expr) for expr in g2_exprs])

        cons = [
            f1_sum >= (tau / omega1 + 1.0) * R_star * np.log(2.0),   # Eq. 73b
            g2_sum >= (tau / omega2 + 1.0) * I_star * np.log(2.0),   # Eq. 73c
            cp.sum_squares(F @ W_var) <= Pt,                           # Eq. 73e
        ]
        # Eq. 63d: domain constraint d_k^W >= eps_dom
        for expr in g2_exprs:
            cons.append(expr >= eps_dom - 1.0)

        prob = cp.Problem(cp.Maximize(tau), cons)
        try:
            prob.solve(solver=alg_cfg.cvx_solver, verbose=alg_cfg.cvx_verbose)
        except Exception:
            prob.solve(solver="SCS", verbose=False)

        if W_var.value is None:
            break                                  # CVX failed; use best W so far

        W_new   = scale_W_to_power(F, np.asarray(W_var.value), Pt)
        tau_new = float(tau.value) if tau.value is not None else tau_val

        # line 9: stopping |tau_W^(i) - tau_W^(i-1)| <= tol
        if tau_val is not None and abs(tau_new - tau_val) < sp5_tol:
            W_curr  = W_new
            tau_val = tau_new
            success = True
            break

        tau_val = tau_new
        W_curr  = W_new
    #     success = True

    # if not success:
    #     return _sp5_fallback(scenario, sys_cfg, alg_cfg, a, W_prev,
    #                          omega1, omega2, R_star, I_star)
    return W_curr, tau_val


# =====================================================================
# SP6 : amplitude BF via Lagrangian dual  — Algorithm 1 lines 14-29
# =====================================================================
def _solve_SP6(scenario, sys_cfg, alg_cfg, a_init, W, R_star, I_star,
               omega1, omega2,
               lam1_init: float = None, lam2_init: float = None,
               full_history: bool = False, sp5_test: bool = False):
    """
    Amplitude sub-problem with fixed W (Algorithm 1 lines 14-29).

    Inner loop order (lines 19-22):
      1. Update mu*, xi*  (Eq. 49-50)
      2. Set z = a^(t-1)
      3. Update lambda1, lambda2 FIRST  (Eq. 76-78)
      4. Compute gradient of psi wrt a using UPDATED lambdas  (Eq. 79)
      5. Update a via BTLS projected gradient  (Eq. 80)
      6. Recover tau^(t)  (Eq. 81)

    Stopping: delta_a <= tol AND delta_in <= tol  (line 25).
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

    # lambda1 initialisation: passed from outer loop (Algorithm 1 line 16)
    lam1_max = omega1 / R_star
    # lam1 = lam1_init if lam1_init is not None else 0.5 * lam1_max
    lam1 = np.random.uniform(0.0, lam1_max) if lam1_init is None else lam1_init
    
    lam2 = (omega2 / I_star) * (1.0 - lam1 * R_star / omega1)
    lam2 = max(lam2, 0.0)

    # best-iterate tracking (safeguard for non-monotone steps)
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

    # pre-compute pieces that don't depend on a:
    # Q_k and coefficient for each ST (sensing surrogate Q matrix)
    Pt = sys_cfg.Pt
    PhiW = Phi @ W                                 # (Mt, Kc+Ks)
    PhiW_PhiW_H = PhiW @ PhiW.conj().T             # (Mt, Mt)

    # P_W = diag(Re{diag(C_W)}),  p_m = ||[Phi W]_m||^2  (Eq. 74-75)
    # Power constraint: a^T P_W a <= Pt  (Eq. 76)
    p_diag = np.real(np.diag(PhiW_PhiW_H))         # (Mt,), non-negative

    Q_sens = []
    coeff_sens = []
    for k in range(Ks):
        bk = scenario.Bt_s[:, k]
        Qk = np.conj(bk)[:, None] * PhiW_PhiW_H * bk[None, :]
        Q_sens.append(Qk)
        coeff_sens.append((L * Mr * scenario.gamma_s2[k]) / sigma_s2)

    bt_beta = getattr(alg_cfg, "bt_beta", 0.5)
    bt_max  = getattr(alg_cfg, "bt_max", 20)
    if sp5_test:
        print("[**********SP5 Testing**********]: SP6 is not solved]")
        best_tau_sp6  = _tau_sp6(R0_sp6, I0_sp6)
        best_a_sp6    = np.random.uniform(0.0, 1.0, size=Mt)
        best_lam1_sp6 = lam1
        best_lam2_sp6 = lam2
    else:
        # inner loop: lines 18-27 of Algorithm 1
        for t in range(alg_cfg.MOOP_inner_iters):

            # ---- Step 1: eval R, I at current a^(t-1) ----
            F = compute_F(a, Phi)
            HF = H @ F
            R_cur = sum_rate(scenario.H, F, W, sigma2, Kc)
            I_cur = sensing_mi(scenario.Bt_s, scenario.gamma_s2, F, W,
                            sigma_s2, sys_cfg.L, Mr)
            tau_cur = _tau_sp6(R_cur, I_cur)

            # ---- Step 2 (line 19): update mu*, xi* (Eq. 49-50) ----
            sig  = HF @ W
            abs2 = np.abs(sig) ** 2
            mu   = np.zeros(Kc)
            xi   = np.zeros(Kc, dtype=complex)
            for k in range(Kc):
                num    = abs2[k, k]
                denom  = abs2[k, :].sum() - abs2[k, k] + sigma2
                mu[k]  = num / max(denom, 1e-30)
                denom2 = abs2[k, :].sum() + sigma2
                xi[k]  = (np.sqrt(1 + mu[k]) * (HF[k] @ W[:, k])
                        / max(denom2, 1e-30))

            # ---- Step 3 (line 20): z = a (linearization point) ----
            z = a.copy()

            # ---- Step 4 (line 21): UPDATE lambda1, lambda2 FIRST (Eq. 76-78) ----
            # dg/dlambda1 (Eq. 77), evaluated at current a (tight surrogate)
            d_lam1 = ((R_cur - R_star)
                    - (omega2 * R_star) / (omega1 * I_star) * (I_cur - I_star))
            lam1 = float(np.clip(lam1 - alg_cfg.pgd_step_lambda * d_lam1,
                                0.0, lam1_max))                # Eq. 76
            # lam2 = max((omega2 / I_star) * (1.0 - lam1 * R_star / omega1), 0.0)  # Eq. 78
            lam2 = (omega2 / I_star) * (1.0 - lam1 * R_star / omega1)
            # ---- Step 5: build Q_bar (Eq. 71), u_bar (Eq. 72) for comm surrogate ----
            Q_bar = np.zeros((Mt, Mt))
            u_bar = np.zeros(Mt)
            for k in range(Kc):
                Dk    = H[k, :]
                DkPhi = Dk[:, None] * Phi
                Qk    = DkPhi @ W
                Q_bar += (np.abs(xi[k]) ** 2) * np.real(Qk @ Qk.conj().T)
                u_bar += 2.0 * np.sqrt(1 + mu[k]) * np.real(
                    (DkPhi @ W[:, k]) * np.conj(xi[k]))

            # ---- gradient of psi wrt a using UPDATED lam1, lam2 (Eq. 79) ----
            grad_comm = lam1 * (-2.0 * (Q_bar @ a) + u_bar)

            grad_sens = np.zeros(Mt)
            for k in range(Ks):
                ck  = coeff_sens[k]
                # Qk  = np.real(ck * Q_sens[k])
                Qk = ck * Q_sens[k]
                Qkz = Qk @ z
                zQz = float(np.real(z @ Qkz))
                zQa = float(np.real(z @ (Qk @ a)))
                denom = 1.0 - zQz + 2.0 * zQa          # = 1 + ck*(2 a^T Qk z - z^T Qk z)
                grad_sens += 2.0 * np.real(Qkz) / max(denom, 1e-12)
            grad_sens *= lam2

            grad_psi  = grad_comm + grad_sens
            grad_norm = float(np.linalg.norm(grad_psi))

            # ---- Step 6 (line 22): BTLS update a^(t) by Eq. (80) ----
            a_prev = a.copy()
            step   = alg_cfg.pgd_step_a / (1.0 + grad_norm / max(Mt, 1))
            R_now, I_now, tau_now = R_cur, I_cur, tau_cur
            for _bt in range(bt_max):
                # Project onto A_W = {a: 0<=a_m<=1, a^T P_W a <= Pt}  (Eq. 80)
                # Step 1: box projection
                a_cand = np.clip(a_prev + step * grad_psi, 0.0, 1.0)
                # Step 2: if power constraint violated, scale down uniformly
                pwr_cand = float(np.dot(p_diag, a_cand ** 2))
                if pwr_cand > Pt:
                    a_cand *= np.sqrt(Pt / pwr_cand)
                F_cand  = compute_F(a_cand, Phi)
                R_cand  = sum_rate(scenario.H, F_cand, W, sigma2, Kc)
                I_cand  = sensing_mi(scenario.Bt_s, scenario.gamma_s2, F_cand, W,
                                    sigma_s2, sys_cfg.L, Mr)
                tau_cand = _tau_sp6(R_cand, I_cand)
                if tau_cand >= tau_cur - 1e-12:
                    a = a_cand
                    R_now, I_now, tau_now = R_cand, I_cand, tau_cand
                    break
                step *= bt_beta

            da_norm = float(np.linalg.norm(a - a_prev))

            # ---- Step 7 (line 23): recover tau^(t) by Eq. (81) ----
            # Δ_in = |tau^(t) - tau^(t-1)|,  Δ_a = ||a^(t) - a^(t-1)||
            delta_in = abs(tau_now - tau_cur)
            delta_a  = da_norm

            # best-iterate safeguard (under monotone BTLS, best == latest accepted)
            if tau_now > best_tau_sp6:
                best_tau_sp6  = tau_now
                best_a_sp6    = a.copy()
                best_lam1_sp6 = lam1
                best_lam2_sp6 = lam2

            if full_history:
                inner_hist["R"].append(R_now)
                inner_hist["I"].append(I_now)
                inner_hist["duality_gap"].append(delta_in)
                inner_hist["lam1"].append(lam1)
                inner_hist["lam2"].append(lam2)
                inner_hist["grad_norm"].append(grad_norm)
                inner_hist["da_norm"].append(da_norm)

            # ---- line 25: stop if Δ_a <= tol AND Δ_in <= tol ----
            if delta_a <= alg_cfg.tol and delta_in <= alg_cfg.tol:
                break

    return best_a_sp6, best_lam1_sp6, best_lam2_sp6, inner_hist


# =====================================================================
# Top-level MOOP runner  — Algorithm 1
# =====================================================================
def solve_MOOP(scenario, sys_cfg, alg_cfg,
               omega1: float = None, omega2: float = None,
               R_star: float = None, I_star: float = None,
               soop1_result=None, soop2_result=None,
               full_history: bool = False):
    """
    Run Algorithm 1 (MOOP). If R*, I* are not provided, run SOOP1/SOOP2
    first to get them.

    Returns dict with W, a, history including R, I and tau per iter.
    """
    if omega1 is None:
        omega1 = alg_cfg.omega1
    if omega2 is None:
        omega2 = alg_cfg.omega2

    # ---- get reference points if needed (Sec. V-C reference generation) ----
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

    R_star_safe = max(R_star, 1e-9)
    I_star_safe = max(I_star, 1e-9)

    # Algorithm 1 line 2: Initialize A(0) ~ Unif[0,1], W(0) ~ Unif[0,1]
    a = rng.uniform(0.0, 1.0, size=Mt)
    F = compute_F(a, Phi)

    ############## W initialization: if SOOP results provided, use their W as a warm-start;
    if soop1_result is not None and soop2_result is not None:
        W = 0.5 * soop1_result["W"] + 0.5 * soop2_result["W"]
    elif soop1_result is not None:
        W = soop1_result["W"]
    elif soop2_result is not None:
        W = soop2_result["W"]
    else:
        W = (rng.standard_normal((N, Kc + Ks))
             + 1j * rng.standard_normal((N, Kc + Ks))) / np.sqrt(2)
    W = scale_W_to_power(F, W, Pt)

    history = {"sum_rate": [], "sensing_mi": [], "tau": []}
    if full_history:
        history["inner_R"] = []
        history["inner_I"] = []
        history["inner_tau"] = []
        history["inner_duality_gap"] = []
        history["sp6_lam1"] = []
        history["sp6_lam2"] = []
        history["sp6_grad_norm"] = []
        history["sp6_da_norm"] = []

    ########### initial tau^(0)##############
    def _tau(R, I):
        return min(omega1 * (R - R_star_safe) / R_star_safe,
                   omega2 * (I - I_star_safe) / I_star_safe)

    
    R0 = sum_rate(scenario.H, F, W, sys_cfg.sigma2, Kc)
    I0 = sensing_mi(scenario.Bt_s, scenario.gamma_s2, F, W,
                    sys_cfg.sigma_s2, sys_cfg.L, scenario.Mr)
    tau_cur = _tau(R0, I0)
    best_tau = tau_cur
    best_a, best_W = a.copy(), W.copy()
    best_R, best_I = R0, I0

    # lam1 warm-start: centre of feasible range; updated only on acceptance
    lam1_warm = 0.5 * omega1 / R_star_safe
    no_progress = 0
    # outer_patience = getattr(alg_cfg, 'MOOP_outer_patience', 5)
    outer_patience = alg_cfg.MOOP_outer_patience
    # outer loop: BCD order SP5 → SP6
    for it in range(alg_cfg.MOOP_outer_iters):
        a_prev_iter   = a.copy()
        W_prev_iter   = W.copy()
        tau_prev_iter = tau_cur
        lam1_warm_prev = lam1_warm

        # ---- SP5: digital BF for current a ----
        W_new, _ = _solve_SP5(scenario, sys_cfg, alg_cfg,
                               a, W, R_star, I_star, omega1, omega2,
                               outer_iter=it, tau_init=tau_cur)

        # full_history checkpoint: state after SP5, before SP6
        if full_history:
            F_ck = compute_F(a, Phi)
            R_ck = sum_rate(scenario.H, F_ck, W_new, sys_cfg.sigma2, Kc)
            I_ck = sensing_mi(scenario.Bt_s, scenario.gamma_s2, F_ck, W_new,
                              sys_cfg.sigma_s2, sys_cfg.L, scenario.Mr)

        # ---- SP6: amplitude update for W_new ----
        a_new, lam1_best, _, sp6_hist = _solve_SP6(
            scenario, sys_cfg, alg_cfg,
            a, W_new, R_star, I_star,
            omega1, omega2, lam1_init=lam1_warm,
            full_history=full_history)
        F_new = compute_F(a_new, Phi)
        W_new = scale_W_to_power(F_new, W_new, sys_cfg.Pt)

        if full_history and sp6_hist is not None:
            outer_R   = [R_ck] + sp6_hist["R"]
            outer_I   = [I_ck] + sp6_hist["I"]
            outer_gap = [float("nan")] + sp6_hist["duality_gap"]
            history["inner_R"].append(outer_R)
            history["inner_I"].append(outer_I)
            history["inner_tau"].append([_tau(r, i) for r, i in zip(outer_R, outer_I)])
            history["inner_duality_gap"].append(outer_gap)
            history["sp6_lam1"].append(sp6_hist["lam1"])
            history["sp6_lam2"].append(sp6_hist["lam2"])
            history["sp6_grad_norm"].append(sp6_hist["grad_norm"])
            history["sp6_da_norm"].append(sp6_hist["da_norm"])

        # evaluate the candidate (a_new, W_new)
        R_cand = sum_rate(scenario.H, F_new, W_new, sys_cfg.sigma2, Kc)
        I_cand = sensing_mi(scenario.Bt_s, scenario.gamma_s2, F_new, W_new,
                            sys_cfg.sigma_s2, sys_cfg.L, scenario.Mr)
        tau_cand = _tau(R_cand, I_cand)

        # ---- outer acceptance: accept only if tau does not decrease ----
        if tau_cand >= best_tau or it == 0:
            a, W = a_new, W_new
            tau_cur = tau_cand
            best_tau = tau_cur
            best_a, best_W = a.copy(), W.copy()
            best_R, best_I = R_cand, I_cand
            lam1_warm = lam1_best
            R_now, I_now = R_cand, I_cand
            accepted = True
            no_progress = 0
        else:
            a, W = a_prev_iter, W_prev_iter
            lam1_warm = lam1_warm_prev
            tau_cur = tau_prev_iter
            F_rev = compute_F(a, Phi)
            R_now = sum_rate(scenario.H, F_rev, W, sys_cfg.sigma2, Kc)
            I_now = sensing_mi(scenario.Bt_s, scenario.gamma_s2, F_rev, W,
                               sys_cfg.sigma_s2, sys_cfg.L, scenario.Mr)
            accepted = False
            no_progress += 1

        print(f"  [MOOP s={it}] R={R_now:.3f} I={I_now:.3f} tau={tau_cur:.4f}"
              f"  dtau={abs(tau_cur - tau_prev_iter):.4f}"
              f"  {'ACC' if accepted else 'REJ'}"
              f"  no_prog={no_progress}")

        history["sum_rate"].append(R_now)
        history["sensing_mi"].append(I_now)
        history["tau"].append(tau_cur)       # non-decreasing by construction

        # stop when accepted tau is flat for outer_patience consecutive iters
        if no_progress >= outer_patience:
            print(f"  [MOOP] outer stopped: {outer_patience} consecutive rejections")
            break

    # always return the best iterate seen
    W_new, _ = _solve_SP5(scenario, sys_cfg, alg_cfg,
                               best_a, W, R_star, I_star, omega1, omega2,
                               outer_iter=it, tau_init=tau_cur)
    a, W = best_a, W_new
    # a, W = best_a, best_W
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
