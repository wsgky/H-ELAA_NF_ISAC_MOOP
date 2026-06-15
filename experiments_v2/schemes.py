"""
Benchmark-scheme registry (manuscript Sec. V-B) used by every sweep
experiment in ``experiments_v2``.

Each scheme is a callable

    scheme(scenario, sys_cfg, alg_cfg, w1, ref) -> (R, I)

where ``ref`` is the dict returned by :func:`build_ref` (the SOOP1/SOOP2
reference point + results for this scenario, computed once per MC trial
and shared across schemes/weights). ``w1`` is the Tchebycheff weight
omega1 (omega2 = 1 - w1); schemes that do not depend on the weight
(``fully_digital``) simply ignore it.

Schemes
-------
proposed        : joint digital + holographic MOOP (Algorithm 1).
fully_digital   : ideal upper bound, BS controls all Mt elements
                  directly (F = I_Mt); R*_fd via ZF+water-filling,
                  I*_fd via per-target matched filtering, equal power.
amplitude_only  : W fixed to ZF + equal power; only `a` is optimised
                  (via the existing SP6 amplitude sub-problem).
random_rhs      : a ~ U(0,1) fixed (seeded); only W is optimised
                  (via the existing SP5 digital sub-problem).
uniform_rhs     : a = 1 (non-optimised aperture); only W is optimised
                  (via the existing SP5 digital sub-problem).
far_field       : design (a, W) on a far-field (planar-wave) copy of
                  the scenario, then evaluate (R, I) on the TRUE
                  near-field channel.

All math is reused from ``algorithms`` / ``system_model``; this file
only assembles existing pieces (ZF+WF, SP5, SP6, MRT, far-field array
response via r -> inf) into the benchmark definitions above.
"""
from dataclasses import replace

import numpy as np

from algorithms.utils import compute_F, sum_rate, sensing_mi, scale_W_to_power
from algorithms.soop1_comm import _zf_waterfilling
from algorithms.moop import _solve_SP5_v2, _solve_SP6
from system_model.channel import near_field_array_response

from ._base import reference_points, solve_proposed, eval_metrics


SCHEME_NAMES = [
    "proposed",
    "fully_digital",
    "amplitude_only",
    "random_rhs",
    "uniform_rhs",
    "far_field",
]

# range (m) used for the far-field / planar-wave approximation: q(r) -> 1
_FAR_FIELD_RANGE_M = 1.0e6


# ----------------------------------------------------------------------
# Reference-point bundle (shared across schemes/weights for one scenario)
# ----------------------------------------------------------------------
def build_ref(scenario, sys_cfg, alg_cfg) -> dict:
    """Run SOOP1/SOOP2 once; bundle (R_star, I_star, soop1, soop2)."""
    R_star, I_star, soop1, soop2 = reference_points(scenario, sys_cfg, alg_cfg)
    return {"R_star": R_star, "I_star": I_star, "soop1": soop1, "soop2": soop2}


# ----------------------------------------------------------------------
# proposed
# ----------------------------------------------------------------------
def _proposed(scenario, sys_cfg, alg_cfg, w1, ref):
    res = solve_proposed(scenario, sys_cfg, alg_cfg, w1,
                          R_star=ref["R_star"], I_star=ref["I_star"],
                          soop1_result=ref["soop1"], soop2_result=ref["soop2"])
    return eval_metrics(scenario, sys_cfg, res["F"], res["W"])


# ----------------------------------------------------------------------
# fully_digital: BS controls all Mt elements directly (no RHS feed
# network), F = I_Mt. R*_fd and I*_fd are independent single-objective
# upper bounds (do not depend on w1).
# ----------------------------------------------------------------------
def _fully_digital(scenario, sys_cfg, alg_cfg, w1, ref):
    Mt, Kc, Ks = scenario.Mt, scenario.Kc, scenario.Ks
    Pt, sigma2 = sys_cfg.Pt, sys_cfg.sigma2
    F_id = np.eye(Mt, dtype=complex)

    # R*_fd: ZF + water-filling with a fully-digital array (F = I_Mt)
    W_c, _ = _zf_waterfilling(scenario.H, F_id, sigma2, Pt)
    W_R = np.concatenate([W_c, np.zeros((Mt, Ks), dtype=complex)], axis=1)
    R_fd = sum_rate(scenario.H, F_id, W_R, sigma2, Kc)

    # I*_fd: per-target matched filter (MRT), equal power split over Ks
    W_s = np.zeros((Mt, Ks), dtype=complex)
    for i in range(Ks):
        g = np.sqrt(scenario.gamma_s2[i]) * scenario.Bt_s[:, i]
        gn = np.linalg.norm(g)
        if gn > 1e-12:
            W_s[:, i] = (g / gn) * np.sqrt(Pt / Ks)
    W_I = np.concatenate([np.zeros((Mt, Kc), dtype=complex), W_s], axis=1)
    I_fd = sensing_mi(scenario.Bt_s, scenario.gamma_s2, F_id, W_I,
                      sys_cfg.sigma_s2, sys_cfg.L, scenario.Mr)
    return float(R_fd), float(I_fd)


# ----------------------------------------------------------------------
# amplitude_only: W fixed to ZF + equal power; optimise `a` only (SP6)
# ----------------------------------------------------------------------
def _zf_equal_power(H, F, Pt, Kc):
    """ZF precoding directions with equal radiated power Pt/Kc per stream."""
    H_bar = H @ F
    HHH = H_bar @ H_bar.conj().T
    HHH_inv = np.linalg.pinv(HHH)
    W_bar = H_bar.conj().T @ HHH_inv          # (N, Kc), H_bar W_bar = I

    FW_bar = F @ W_bar
    c = np.real(np.sum(np.conj(FW_bar) * FW_bar, axis=0))
    c = np.maximum(c, 1e-12)
    p = np.full(Kc, Pt / Kc)
    return W_bar * np.sqrt(p / c)[None, :]


def _amplitude_only(scenario, sys_cfg, alg_cfg, w1, ref):
    Mt, N, Kc, Ks = scenario.Mt, scenario.N, scenario.Kc, scenario.Ks
    Phi = scenario.Phi

    a0 = np.ones(Mt)
    F0 = compute_F(a0, Phi)
    W_c = _zf_equal_power(scenario.H, F0, sys_cfg.Pt, Kc)
    W = np.concatenate([W_c, np.zeros((N, Ks), dtype=complex)], axis=1)
    W = scale_W_to_power(F0, W, sys_cfg.Pt)

    a_opt, _, _, _ = _solve_SP6(scenario, sys_cfg, alg_cfg, a0, W,
                                ref["R_star"], ref["I_star"], w1, 1.0 - w1)
    F = compute_F(a_opt, Phi)
    return eval_metrics(scenario, sys_cfg, F, W)


# ----------------------------------------------------------------------
# random_rhs / uniform_rhs: `a` fixed; optimise W only (SP5)
# ----------------------------------------------------------------------
def _fixed_a_solve_W(scenario, sys_cfg, alg_cfg, a, w1, ref):
    Phi = scenario.Phi
    F = compute_F(a, Phi)

    W_c, _ = _zf_waterfilling(scenario.H, F, sys_cfg.sigma2, sys_cfg.Pt)
    W_init = np.concatenate(
        [W_c, np.zeros((scenario.N, scenario.Ks), dtype=complex)], axis=1)
    W_init = scale_W_to_power(F, W_init, sys_cfg.Pt)

    W_opt, _ = _solve_SP5_v2(scenario, sys_cfg, alg_cfg, a, W_init,
                             ref["R_star"], ref["I_star"], w1, 1.0 - w1,
                             outer_iter=0, tau_init=None)
    return eval_metrics(scenario, sys_cfg, F, W_opt)


def _random_rhs(scenario, sys_cfg, alg_cfg, w1, ref):
    rng = np.random.default_rng(sys_cfg.seed + 9001)
    a = rng.uniform(0.0, 1.0, size=scenario.Mt)
    return _fixed_a_solve_W(scenario, sys_cfg, alg_cfg, a, w1, ref)


def _uniform_rhs(scenario, sys_cfg, alg_cfg, w1, ref):
    a = np.ones(scenario.Mt)
    return _fixed_a_solve_W(scenario, sys_cfg, alg_cfg, a, w1, ref)


# ----------------------------------------------------------------------
# far_field: design on a far-field (planar-wave) copy of the scenario
# (r -> _FAR_FIELD_RANGE_M, same directions), evaluate on the true
# near-field channel.
# ----------------------------------------------------------------------
def _far_field_scenario(scenario):
    """Return a copy of ``scenario`` with H, Bt_*, Br_* replaced by their
    far-field (planar-wave) counterparts (same angles, r -> infinity)."""
    Mt_h, Mt_v = scenario.Mt_h, scenario.Mt_v
    Mr_h, Mr_v = scenario.Mr_h, scenario.Mr_v
    delta, lambda0 = scenario.delta, scenario.lambda0
    Mt = Mt_h * Mt_v
    R = _FAR_FIELD_RANGE_M

    def _ang(pos):
        r = np.linalg.norm(pos)
        phi = np.arctan2(pos[1], pos[0])
        theta = np.arcsin(pos[2] / max(r, 1e-12))
        return phi, theta

    # ---- CUs: rebuild h_k with far-field LoS + NLoS responses ----
    H_ff = np.zeros_like(scenario.H)
    Bt_c_ff = np.zeros_like(scenario.Bt_c)
    Br_c_ff = np.zeros_like(scenario.Br_c)
    for k, info in enumerate(scenario.cu_info):
        phi0, theta0, alpha0 = info["phi0"], info["theta0"], info["alpha0"]
        b0 = near_field_array_response(Mt_h, Mt_v, delta, R, phi0, theta0, lambda0)
        h = np.sqrt(Mt) * alpha0 * b0
        J = len(info["nlos_dirs"])
        for (_rj, phij, thetaj, alphaj) in info["nlos_dirs"]:
            bj = near_field_array_response(Mt_h, Mt_v, delta, R, phij, thetaj, lambda0)
            h = h + np.sqrt(Mt / J) * alphaj * bj
        H_ff[k:k + 1] = h.reshape(1, Mt)

        Bt_c_ff[:, k] = near_field_array_response(Mt_h, Mt_v, delta, R, phi0, theta0, lambda0)
        Br_c_ff[:, k] = near_field_array_response(Mr_h, Mr_v, delta, R, phi0, theta0, lambda0)

    # ---- STs ----
    Bt_s_ff = np.zeros_like(scenario.Bt_s)
    Br_s_ff = np.zeros_like(scenario.Br_s)
    for i, pos in enumerate(scenario.st_positions):
        phi, theta = _ang(pos)
        Bt_s_ff[:, i] = near_field_array_response(Mt_h, Mt_v, delta, R, phi, theta, lambda0)
        Br_s_ff[:, i] = near_field_array_response(Mr_h, Mr_v, delta, R, phi, theta, lambda0)

    # ---- EOs ----
    Bt_e_ff = np.zeros_like(scenario.Bt_e)
    Br_e_ff = np.zeros_like(scenario.Br_e)
    for j, pos in enumerate(scenario.eo_positions):
        phi, theta = _ang(pos)
        Bt_e_ff[:, j] = near_field_array_response(Mt_h, Mt_v, delta, R, phi, theta, lambda0)
        Br_e_ff[:, j] = near_field_array_response(Mr_h, Mr_v, delta, R, phi, theta, lambda0)

    return replace(scenario,
                   H=H_ff, Bt_c=Bt_c_ff, Br_c=Br_c_ff,
                   Bt_s=Bt_s_ff, Br_s=Br_s_ff,
                   Bt_e=Bt_e_ff, Br_e=Br_e_ff)


def _far_field(scenario, sys_cfg, alg_cfg, w1, ref):
    ff_scenario = _far_field_scenario(scenario)
    res = solve_proposed(ff_scenario, sys_cfg, alg_cfg, w1,
                          R_star=ref["R_star"], I_star=ref["I_star"],
                          soop1_result=ref["soop1"], soop2_result=ref["soop2"])
    F_true = compute_F(res["a"], scenario.Phi)
    return eval_metrics(scenario, sys_cfg, F_true, res["W"])


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------
SCHEME_FUNCS = {
    "proposed": _proposed,
    "fully_digital": _fully_digital,
    "amplitude_only": _amplitude_only,
    "random_rhs": _random_rhs,
    "uniform_rhs": _uniform_rhs,
    "far_field": _far_field,
}


def run_scheme(name: str, scenario, sys_cfg, alg_cfg, w1: float, ref: dict):
    """Evaluate scheme ``name`` -> (R, I)."""
    return SCHEME_FUNCS[name](scenario, sys_cfg, alg_cfg, w1, ref)
