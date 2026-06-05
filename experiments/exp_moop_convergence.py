"""
MOOP fine-grained convergence analysis.

For each outer iteration the inner SP6 loop runs up to `inner_iters` steps
(early-stopped by the duality-gap criterion). This script draws two figures:

For a weighted-Tchebycheff MOOP, R and I trade off against each other, so
they need NOT both increase monotonically. The quantity that MUST be monotone
non-decreasing is the scalarised objective tau; the accepted outer-level tau
curve is therefore the real convergence-proof figure.

Figure 1 — 2×2 coarse convergence (saved as exp_moop_convergence_<ts>.png)
  Panel [0,0] R (sum-rate)       : SP5 checkpoint + all SP6 inner steps
  Panel [0,1] I (sensing MI)     : same
  Panel [1,0] tau (Tchebycheff) : min(omega1*(R-R*)/R*, omega2*(I-I*)/I*);
                                   accepted outer-level tau drawn as a step
                                   line — this is the curve that must rise.
  Panel [1,1] Optimality gap (log) : per-step improvement tau^(k)-tau^(k-1)>=0
                                      of the monotone SP6 inner loop; -> 0.

Figure 2 — 2×4 SP6 per-outer-iter detail (saved as ..._sp6detail_<ts>.png)
  Row 0: R, I, tau, optimality-gap Δτ vs SP6 inner step (one line / outer iter)
  Row 1: lambda_1, lambda_2, ||grad_a L||, ||Delta a|| vs SP6 inner step

Run:
    python -m experiments.exp_moop_convergence
"""
import json
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")          # save to PNG; open with os.startfile below
import matplotlib.pyplot as plt
from pathlib import Path

from config import SystemConfig, AlgorithmConfig
from experiments.runner import RESULTS_DIR, _to_jsonable
from system_model import generate_scenario
from algorithms import solve_SOOP1, solve_SOOP2, solve_MOOP


# ── small config: runs fast enough to see convergence clearly ─────────────
_SYS_CFG = SystemConfig(
    Mt_h=128, Mt_v=8,    # Mt = 128
    Mr_h=128, Mr_v=8,    # Mr = 128
    N=8,
    Kc=3, Ks=3, Ke=2,
    L=128,
    seed=2025,
    rhs_structure= "fully_connected"
)
_ALG_CFG = AlgorithmConfig(
    SOOP1_outer_iters=20,
    SOOP2_outer_iters=20,
    MOOP_outer_iters=25,
    SOOP1_inner_iters=500,
    SOOP2_inner_iters=500,
    MOOP_inner_iters=500,
    sp5_iters=20,
    sp5_tol=1e-4,
    pgd_step_a=1e-2,
    pgd_step_lambda=8e-2,
    tol=1e-5,
    # SP6 monotonic backtracking line search on the original tau
    bt_beta=0.5,
    bt_max=20,
    MOOP_inner_patience=20,
    omega1=0.5,
    omega2=0.5)


# ─────────────────────────────────────────────────────────────────────────────
def plot_moop_convergence(m, R_star, I_star, omega1, omega2, alg_cfg,
                          suptitle=""):
    """
    Build the 2×2 figure from a MOOP result produced with full_history=True.

    Panels:
      [0,0] R convergence       [0,1] I convergence
      [1,0] τ convergence       [1,1] Duality gap (log scale)

    Returns the matplotlib Figure.
    """
    hist = m["history"]
    inner_R_all   = hist.get("inner_R",   [])
    inner_I_all   = hist.get("inner_I",   [])
    inner_tau_all = hist.get("inner_tau", [])
    inner_gap_all = hist.get("inner_duality_gap", [])

    if not inner_R_all:
        raise ValueError("MOOP result was not run with full_history=True")

    n_outer = len(inner_R_all)

    # ── unroll fine-grained curves ────────────────────────────────────────
    fine_R   = [v for seg in inner_R_all   for v in seg]
    fine_I   = [v for seg in inner_I_all   for v in seg]
    fine_tau = [v for seg in inner_tau_all for v in seg]
    x_fine   = np.arange(len(fine_R))

    # Duality-gap curve: each outer iter may have different length (early stop).
    gap_x, gap_y = [], []
    cursor = 0
    for seg in inner_gap_all:
        for k, v in enumerate(seg):
            if not np.isnan(v):
                gap_x.append(cursor + k)
                gap_y.append(v)
        cursor += len(seg)

    # outer-iter boundary x-positions (last index of each outer iter's segment)
    outer_x = []
    cursor = 0
    for seg in inner_R_all:
        cursor += len(seg)
        outer_x.append(cursor - 1)
    outer_x = np.array(outer_x)

    # accepted outer-iter values (coarse history)
    outer_R   = hist["sum_rate"][:n_outer]
    outer_I   = hist["sensing_mi"][:n_outer]
    outer_tau = hist["tau"][:n_outer]

    # ── figure : 2×2 layout ───────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    clr_fine  = "C0"
    clr_outer = "C1"
    clr_ref   = "C2"
    clr_gap   = "C3"
    marker_kw = dict(marker="D", s=60, zorder=5, edgecolors="k", linewidths=0.5)

    def _vlines(ax, total_len):
        cursor = 0
        for k, seg in enumerate(inner_R_all):
            cursor += len(seg)
            if k < n_outer - 1:
                ax.axvline(cursor - 0.5, color="gray", linestyle="--",
                           linewidth=0.7, alpha=0.45)
            mid = (cursor - len(seg) / 2) / max(total_len - 1, 1)
            ax.text(mid, 0.99, f"It.{k}", ha="center", va="top",
                    fontsize=6.0, color="gray", transform=ax.transAxes)

    # ── [0,0] R ───────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(x_fine, fine_R, color=clr_fine, linewidth=1.2, label="SP6 inner steps")
    ax.scatter(outer_x, outer_R, color=clr_outer, label="Accepted outer-iter R",
               **marker_kw)
    ax.axhline(R_star, color=clr_ref, linestyle=":", linewidth=1.6,
               label=f"R* = {R_star:.3f}")
    ax.set_xlabel("Cumulative inner step")
    ax.set_ylabel("Sum-rate [bits/Hz]")
    ax.set_title("R convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _vlines(ax, len(fine_R))

    # ── [0,1] I ───────────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(x_fine, fine_I, color=clr_fine, linewidth=1.2, label="SP6 inner steps")
    ax.scatter(outer_x, outer_I, color=clr_outer, label="Accepted outer-iter I",
               **marker_kw)
    ax.axhline(I_star, color=clr_ref, linestyle=":", linewidth=1.6,
               label=f"I* = {I_star:.3f}")
    ax.set_xlabel("Cumulative inner step")
    ax.set_ylabel("Sensing MI [bits/Hz]")
    ax.set_title("I convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _vlines(ax, len(fine_I))

    # ── [1,0] tau  (the curve that MUST be monotone) ──────────────────────
    ax = axes[1, 0]
    ax.plot(x_fine, fine_tau, color=clr_fine, linewidth=1.0, alpha=0.5,
            label="SP6 inner steps")
    # accepted outer-level tau: this is the convergence-proof curve
    ax.plot(outer_x, outer_tau, color=clr_outer, linewidth=1.8,
            marker="D", markersize=7, markeredgecolor="k", markeredgewidth=0.5,
            zorder=6, label="Accepted outer-iter τ")
    ax.axhline(0, color="gray", linewidth=0.6, alpha=0.5)
    # annotate monotonicity of the accepted outer-level tau
    n_viol = sum(1 for i in range(1, len(outer_tau))
                 if outer_tau[i] < outer_tau[i - 1] - 1e-9)
    ax.text(0.02, 0.02,
            f"accepted-τ monotone: {'YES' if n_viol == 0 else f'NO ({n_viol})'}",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
            color="green" if n_viol == 0 else "red",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))
    ax.set_xlabel("Cumulative inner step")
    ax.set_ylabel("τ  (Tchebycheff objective)")
    ax.set_title("τ convergence (accepted outer-level must be non-decreasing)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _vlines(ax, len(fine_tau))

    # ── [1,1] Optimality gap Δτ (log scale, non-negative) ─────────────────
    # gap = tau^(k) - tau^(k-1) >= 0  by monotone acceptance; -> 0 at a
    # stationary point. Zeros (plateau steps) are clamped to a floor so the
    # log axis stays readable.
    ax = axes[1, 1]
    gap_floor = max(alg_cfg.tol * 1e-3, 1e-12)
    if gap_y:
        gap_y_clamped = [max(v, gap_floor) for v in gap_y]
        ax.semilogy(gap_x, gap_y_clamped, color=clr_gap, linewidth=1.2,
                    label=r"$\Delta\tau^{(k)} = \tau^{(k)} - \tau^{(k-1)} \geq 0$")
        ax.axhline(alg_cfg.tol, color="k", linestyle="--", linewidth=1.2,
                   label=f"tol = {alg_cfg.tol:.0e}")
        for k, seg in enumerate(inner_gap_all):
            sp6_gaps = [v for v in seg if not np.isnan(v)]
            if sp6_gaps:
                seg_start = sum(len(inner_R_all[j]) for j in range(k))
                stop_x = seg_start + len(sp6_gaps)
                ax.scatter([stop_x - 1], [max(sp6_gaps[-1], gap_floor)],
                           color=clr_outer, marker="D", s=55, zorder=5,
                           edgecolors="k", linewidths=0.5)
    ax.set_xlabel("Cumulative inner step (SP6 only)")
    ax.set_ylabel(r"Optimality gap $\Delta\tau$")
    ax.set_title(r"Monotone improvement gap: $\Delta\tau^{(k)} \to 0$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    _vlines(ax, len(fine_R))

    title = suptitle or (
        f"MOOP fine-grained convergence | ω=({omega1:.2f},{omega2:.2f}) | "
        f"{n_outer} outer iters | SP6 monotonic backtracking on original τ "
        f"(stop after {getattr(alg_cfg, 'MOOP_inner_patience', '?')} no-progress steps)"
    )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
def plot_sp6_per_outer(m, R_star, I_star, omega1, omega2, alg_cfg,
                       suptitle=""):
    """
    2×4 figure. One coloured line per outer iteration (tab10 palette).
    Row 0: R, I, τ, optimality-gap Δτ vs SP6 inner step index
    Row 1: λ₁, λ₂, ‖∇_a L‖, ‖Δa‖ vs SP6 inner step index

    The SP5 checkpoint (index 0 of inner_R/I/tau) is stripped; only the
    SP6 steps (index 1 onward) are plotted so the x-axis is the SP6 step.
    """
    hist = m["history"]
    n_outer = len(hist.get("sp6_lam1", []))
    if n_outer == 0:
        raise ValueError("MOOP result does not contain SP6 detail "
                         "(run with full_history=True).")

    # Strip the SP5 checkpoint (first element) from inner_R/I/tau/gap
    sp6_R   = [seg[1:] for seg in hist["inner_R"]]
    sp6_I   = [seg[1:] for seg in hist["inner_I"]]
    sp6_tau = [seg[1:] for seg in hist["inner_tau"]]
    sp6_gap = [[v for v in seg[1:] if not np.isnan(v)]
                for seg in hist["inner_duality_gap"]]
    sp6_lam1 = hist["sp6_lam1"]
    sp6_lam2 = hist["sp6_lam2"]
    sp6_grad = hist["sp6_grad_norm"]
    sp6_da   = hist["sp6_da_norm"]

    cmap = matplotlib.colormaps.get_cmap("tab10")
    colors = [cmap(k % 10) for k in range(n_outer)]

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))

    panel_data = [
        # (row, col, y_data_list, ylabel, title, log_scale)
        (0, 0, sp6_R,    "Sum-rate [bits/Hz]",      "R  vs SP6 step",         False),
        (0, 1, sp6_I,    "Sensing MI [bits/Hz]",     "I  vs SP6 step",         False),
        (0, 2, sp6_tau,  "τ (Tchebycheff)",          "τ  vs SP6 step",         False),
        (0, 3, sp6_gap,  r"Optimality gap $\Delta\tau$", "Δτ optimality gap (log)", True),
        (1, 0, sp6_lam1, "λ₁",                       "λ₁ vs SP6 step",         False),
        (1, 1, sp6_lam2, "λ₂",                       "λ₂ vs SP6 step",         False),
        (1, 2, sp6_grad, r"$\|\nabla_a \mathcal{L}\|$",
                                                      "‖∇_a L‖ vs SP6 step",   True),
        (1, 3, sp6_da,   r"$\|\Delta a\|$",           "‖Δa‖ vs SP6 step",      True),
    ]

    for row, col, data_list, ylabel, title, logscale in panel_data:
        ax = axes[row, col]
        for k, seg in enumerate(data_list):
            if len(seg) == 0:
                continue
            xs = np.arange(len(seg))
            if logscale:
                pos = [max(v, 1e-30) for v in seg]
                ax.semilogy(xs, pos, color=colors[k], linewidth=1.2,
                            label=f"Outer {k}")
            else:
                ax.plot(xs, seg, color=colors[k], linewidth=1.2,
                        label=f"Outer {k}")

        # reference lines
        if col == 0 and row == 0:
            ax.axhline(R_star, color="gray", linestyle=":", linewidth=1.0,
                       label=f"R*={R_star:.2f}")
        if col == 1 and row == 0:
            ax.axhline(I_star, color="gray", linestyle=":", linewidth=1.0,
                       label=f"I*={I_star:.2f}")
        if col == 3 and row == 0 and alg_cfg is not None:
            ax.axhline(alg_cfg.tol, color="k", linestyle="--", linewidth=1.0,
                       label=f"tol={alg_cfg.tol:.0e}")

        ax.set_xlabel("SP6 inner step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, which="both" if logscale else "major")
        if n_outer <= 10:
            ax.legend(fontsize=6, ncol=2)

    title_str = suptitle or (
        f"SP6 per-outer-iter detail | ω=({omega1:.2f},{omega2:.2f}) | "
        f"{n_outer} outer iters | inner_iters={alg_cfg.MOOP_inner_iters if alg_cfg else '?'}"
    )
    fig.suptitle(title_str, fontsize=10)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
def main():
    sys_cfg = _SYS_CFG
    alg_cfg = _ALG_CFG
    rng = np.random.default_rng(sys_cfg.seed)
    scen = generate_scenario(sys_cfg, rng)

    print(f"[exp_moop_convergence] Mt={sys_cfg.Mt}  Mr={sys_cfg.Mr}  "
          f"N={sys_cfg.N}  Kc={sys_cfg.Kc}  Ks={sys_cfg.Ks}")

    print("[exp_moop_convergence] Running SOOP1 ...")
    s1 = solve_SOOP1(scen, sys_cfg, alg_cfg)
    R_star = s1["history"]["sum_rate"][-1]
    print(f"  R* = {R_star:.4f} bits/Hz")

    print("[exp_moop_convergence] Running SOOP2 ...")
    s2 = solve_SOOP2(scen, sys_cfg, alg_cfg)
    I_star = s2["history"]["sensing_mi"][-1]
    print(f"  I* = {I_star:.4f} bits/Hz")

    print("[exp_moop_convergence] Running MOOP (full_history=True) ...")
    t0 = time.time()
    m = solve_MOOP(scen, sys_cfg, alg_cfg,
                   soop1_result=s1, soop2_result=s2,
                   full_history=True)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Final: R={m['history']['sum_rate'][-1]:.4f}  "
          f"I={m['history']['sensing_mi'][-1]:.4f}  "
          f"tau={m['history']['tau'][-1]:.4f}")

    # ── monotonicity check ────────────────────────────────────────────────
    tau_hist = m["history"]["tau"]
    viols = sum(1 for i in range(1, len(tau_hist)) if tau_hist[i] < tau_hist[i-1] - 1e-9)
    print(f"  Tau monotonicity violations (outer): {viols}")

    # ── save JSON ─────────────────────────────────────────────────────────
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = {
        "label": "exp_moop_convergence",
        "sys_cfg": {k: v for k, v in vars(sys_cfg).items()
                    if not k.startswith("_")},
        "alg_cfg": vars(alg_cfg),
        "R_star": R_star,
        "I_star": I_star,
        "omega1": alg_cfg.omega1,
        "omega2": alg_cfg.omega2,
        "MOOP_history": m["history"],
        "SOOP1_history": s1["history"],
        "SOOP2_history": s2["history"],
    }
    json_path = RESULTS_DIR / f"exp_moop_convergence_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(_to_jsonable(out), f, indent=2)
    print(f"  JSON -> {json_path}")

    # ── Figure 1: 2×2 coarse convergence ─────────────────────────────────
    fig1 = plot_moop_convergence(
        m, R_star, I_star, alg_cfg.omega1, alg_cfg.omega2, alg_cfg)
    fig1_path = json_path.with_suffix(".png")
    fig1.savefig(fig1_path, dpi=150, bbox_inches="tight")
    print(f"  Figure 1 -> {fig1_path}")
    plt.close(fig1)
    if sys.platform == "win32":
        os.startfile(str(fig1_path))

    # ── Figure 2: 2×4 SP6 per-outer-iter detail ───────────────────────────
    fig2 = plot_sp6_per_outer(
        m, R_star, I_star, alg_cfg.omega1, alg_cfg.omega2, alg_cfg)
    fig2_path = json_path.with_name(
        json_path.stem + "_sp6detail" + ".png")
    fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
    print(f"  Figure 2 -> {fig2_path}")
    plt.close(fig2)
    if sys.platform == "win32":
        os.startfile(str(fig2_path))


if __name__ == "__main__":
    main()
