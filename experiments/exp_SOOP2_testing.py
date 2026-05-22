"""
SOOP2 fine-grained convergence validation via Monte Carlo.

For every trial the convergence curve is unrolled as:
    outer iter 0 : [I_after_SDP, I_inner_1, ..., I_inner_T]
    outer iter 1 : [I_after_SDP, I_inner_1, ..., I_inner_T]
    ...
giving  outer_iters x (1 + inner_iters)  points per trial.

Usage:
    python main.py SOOP2_testing            # default 20 trials
    python main.py SOOP2_testing 50
    python experiments/exp_SOOP2_testing.py 20
"""
import copy
import json
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from config import AlgorithmConfig, SystemConfig
from experiments.runner import RESULTS_DIR, _to_jsonable
from system_model import generate_scenario
from algorithms import solve_SOOP2
from algorithms.utils import compute_F, sensing_mi, sum_rate


def _isotropic_W(F: np.ndarray, N: int, target_dim: int, Pt: float
                 ) -> np.ndarray:
    """Identity-like W scaled to Pt."""
    W = np.eye(N, target_dim, dtype=complex)
    FW = F @ W
    p_now = float(np.real(np.trace(FW @ FW.conj().T)))
    if p_now > 1e-12:
        W *= np.sqrt(Pt / p_now)
    return W


def _baseline_metrics(scen, cfg, a: np.ndarray):
    """Sensing MI and sum-rate for a given amplitude vector with isotropic W."""
    F = compute_F(a, scen.Phi)
    W = _isotropic_W(F, scen.N, scen.Kc + scen.Ks, cfg.Pt)
    I = sensing_mi(scen.Bt_s, scen.gamma_s2, F, W,
                   cfg.sigma_s2, cfg.L, scen.Mr)
    R = sum_rate(scen.H, F, W, cfg.sigma2, scen.Kc)
    return float(I), float(R)


def main(n_trials: int = 20):
    sys_cfg = SystemConfig(
        Mt_h=128, Mt_v=8, Mr_h=128, Mr_v=8, N=8,
        Kc=3, Ks=2, Ke=2, L=128, seed=2025,
        rhs_structure="subarray",
    )
    alg_cfg = AlgorithmConfig(
        outer_iters=5,
        inner_iters=200,
        pgd_step_a=5e-4,
    )

    OI = alg_cfg.outer_iters
    steps_per_outer = 1 + alg_cfg.inner_iters   # 1 SDP anchor + inner_iters PGD
    total_steps     = OI * steps_per_outer

    print(f"[exp_SOOP2_testing] {n_trials} trials | "
          f"Mt={sys_cfg.Mt_h}x{sys_cfg.Mt_v}  N={sys_cfg.N} "
          f"Kc={sys_cfg.Kc}  Ks={sys_cfg.Ks}  Ke={sys_cfg.Ke} "
          f"Pt={sys_cfg.Pt_dBm} dBm")
    print(f"  outer_iters={OI}  inner_iters={alg_cfg.inner_iters}"
          f"  pgd_step_a={alg_cfg.pgd_step_a}"
          f"  => {total_steps} points/trial")
    print("-" * 65)

    all_I_curves = []
    all_R_curves = []
    final_I_list = []
    final_R_list = []
    I_allon_list, R_allon_list = [], []
    I_rand_list,  R_rand_list  = [], []

    t0 = time.time()

    for trial in range(n_trials):
        seed = sys_cfg.seed + trial + 1
        cfg  = copy.deepcopy(sys_cfg)
        cfg.seed = seed
        rng  = np.random.default_rng(seed)
        scen = generate_scenario(cfg, rng)

        # ── baselines ─────────────────────────────────────────────────────
        I_allon, R_allon = _baseline_metrics(scen, cfg, np.ones(scen.Mt))
        I_allon_list.append(I_allon)
        R_allon_list.append(R_allon)

        a_rand = np.random.default_rng(seed + 500).uniform(0, 1, scen.Mt)
        I_rand, R_rand = _baseline_metrics(scen, cfg, a_rand)
        I_rand_list.append(I_rand)
        R_rand_list.append(R_rand)

        # ── SOOP2 with fine-grained history ───────────────────────────────
        res = solve_SOOP2(scen, cfg, alg_cfg, full_history=True)
        h   = res["history"]

        # unroll: [I_after_SDP, inner_1, ..., inner_T] per outer iter
        I_curve, R_curve = [], []
        for seg_I, seg_R in zip(h["inner_sensing_mi"], h["inner_sum_rate"]):
            I_curve.extend(seg_I)
            R_curve.extend(seg_R)

        def _pad(curve, length):
            if len(curve) >= length:
                return curve[:length]
            return curve + [curve[-1]] * (length - len(curve))

        all_I_curves.append(_pad(I_curve, total_steps))
        all_R_curves.append(_pad(R_curve, total_steps))

        final_I = h["sensing_mi"][-1]
        final_R = h["sum_rate"][-1]
        final_I_list.append(final_I)
        final_R_list.append(final_R)

        mono_ok = all(b >= a - 1e-6
                      for a, b in zip(h["sensing_mi"], h["sensing_mi"][1:]))
        flag = "" if mono_ok else "  [MONO VIOLATION]"
        print(f"  trial {trial+1:3d}/{n_trials} | "
              f"I={final_I:.3f}  R={final_R:.3f}  "
              f"I_allon={I_allon:.3f}  I_rand={I_rand:.3f}{flag}")

    elapsed = time.time() - t0

    I_curves = np.array(all_I_curves)      # (n_trials, total_steps)
    R_curves = np.array(all_R_curves)
    mean_I, std_I = I_curves.mean(axis=0), I_curves.std(axis=0)
    mean_R, std_R = R_curves.mean(axis=0), R_curves.std(axis=0)
    x = np.arange(total_steps)

    I_allon_arr = np.array(I_allon_list);  R_allon_arr = np.array(R_allon_list)
    I_rand_arr  = np.array(I_rand_list);   R_rand_arr  = np.array(R_rand_list)
    I_allon_m, I_allon_s = float(I_allon_arr.mean()), float(I_allon_arr.std())
    I_rand_m,  I_rand_s  = float(I_rand_arr.mean()),  float(I_rand_arr.std())
    R_allon_m, R_allon_s = float(R_allon_arr.mean()), float(R_allon_arr.std())
    R_rand_m,  R_rand_s  = float(R_rand_arr.mean()),  float(R_rand_arr.std())
    soop2_I_m = float(np.mean(final_I_list))
    soop2_I_s = float(np.std(final_I_list))
    soop2_R_m = float(np.mean(final_R_list))
    soop2_R_s = float(np.std(final_R_list))

    # ── summary ────────────────────────────────────────────────────────────
    W = 65
    print("\n" + "=" * W)
    print(f"  {'Algorithm':<22}  {'Mean I':>8}  {'Std I':>7}  {'vs rand-A':>10}")
    print("  " + "-" * (W - 2))
    print(f"  {'SOOP2 (final)':<22}  {soop2_I_m:>8.4f}  {soop2_I_s:>7.4f}"
          f"  {soop2_I_m - I_rand_m:>+10.4f}")
    print(f"  {'All-on (a=1)':<22}  {I_allon_m:>8.4f}  {I_allon_s:>7.4f}"
          f"  {I_allon_m - I_rand_m:>+10.4f}")
    print(f"  {'Random-A':<22}  {I_rand_m:>8.4f}  {I_rand_s:>7.4f}"
          f"  {'(reference)':>10}")
    print("=" * W)
    print(f"\n  Elapsed : {elapsed:.1f} s\n")

    # ── plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    def _fill_plot(ax, mean, std, curves, color):
        for curve in curves:
            ax.plot(x, curve, color=color, alpha=0.12, linewidth=0.7)
        ax.fill_between(x, mean - std, mean + std,
                        alpha=0.30, color=color, label='±1 std')
        ax.plot(x, mean, color=color, linewidth=2.0, label='Mean')

    def _add_baseline(ax, mean_val, std_val, label, color, ls):
        ax.axhline(mean_val, color=color, linestyle=ls, linewidth=1.5,
                   label=f'{label} ({mean_val:.2f})')
        ax.fill_between(x, [mean_val - std_val] * total_steps,
                           [mean_val + std_val] * total_steps,
                        alpha=0.10, color=color)

    def _add_outer_markers(ax):
        for it in range(OI + 1):
            ax.axvline(it * steps_per_outer, color='gray',
                       linestyle='--', linewidth=0.7, alpha=0.5)
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ticks = [it * steps_per_outer + steps_per_outer // 2 for it in range(OI)]
        ax2.set_xticks(ticks)
        ax2.set_xticklabels([f"Out {it}" for it in range(OI)], fontsize=7)
        ax2.tick_params(length=0)

    xlabel = (f"Cumulative step  "
              f"(1 SDP + {alg_cfg.inner_iters} inner PGD  per outer iter)")

    # ---- left: sensing MI (primary) --------------------------------------
    ax = axes[0]
    _fill_plot(ax, mean_I, std_I, all_I_curves, 'C0')
    _add_baseline(ax, I_allon_m, I_allon_s, 'All-on mean', 'C2', '-.')
    _add_baseline(ax, I_rand_m,  I_rand_s,  'Random-A mean', 'C3', ':')
    _add_outer_markers(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Sensing MI [bits/Hz]")
    ax.set_title("Sensing MI convergence  (primary objective)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.35)

    # ---- right: sum-rate (secondary) -------------------------------------
    ax = axes[1]
    _fill_plot(ax, mean_R, std_R, all_R_curves, 'C1')
    _add_baseline(ax, R_allon_m, R_allon_s, 'All-on mean', 'C2', '-.')
    _add_baseline(ax, R_rand_m,  R_rand_s,  'Random-A mean', 'C3', ':')
    _add_outer_markers(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Sum-rate [bits/Hz]")
    ax.set_title("Sum-rate  (secondary — sensing-centric design)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.35)

    cfg_str = (f"Mt={sys_cfg.Mt_h}x{sys_cfg.Mt_v}  N={sys_cfg.N}  "
               f"Kc={sys_cfg.Kc}  Ks={sys_cfg.Ks}  "
               f"Pt={sys_cfg.Pt_dBm} dBm  {sys_cfg.rhs_structure}")
    fig.suptitle(f"SOOP2 Monte Carlo — {n_trials} trials\n{cfg_str}", fontsize=10)
    fig.tight_layout()

    ts = time.strftime("%Y%m%d_%H%M%S")
    png_path = RESULTS_DIR / f"exp_soop2_convergence_{ts}.png"
    fig.savefig(png_path, dpi=150)
    print(f"[exp_SOOP2_testing] figure saved -> {png_path}")

    json_path = RESULTS_DIR / f"exp_soop2_convergence_{ts}.json"
    pack = {
        "label":           "exp_soop2_convergence",
        "n_trials":        n_trials,
        "outer_iters":     OI,
        "inner_iters":     alg_cfg.inner_iters,
        "pgd_step_a":      alg_cfg.pgd_step_a,
        "steps_per_outer": steps_per_outer,
        "sys_cfg": {
            "Mt_h": sys_cfg.Mt_h, "Mt_v": sys_cfg.Mt_v,
            "Mr_h": sys_cfg.Mr_h, "Mr_v": sys_cfg.Mr_v,
            "N": sys_cfg.N, "Kc": sys_cfg.Kc,
            "Ks": sys_cfg.Ks, "Ke": sys_cfg.Ke,
            "L": sys_cfg.L, "Pt_dBm": sys_cfg.Pt_dBm,
            "rhs_structure": sys_cfg.rhs_structure,
        },
        "sensing_mi": {
            "mean_curve":  mean_I.tolist(),
            "std_curve":   std_I.tolist(),
            "all_curves":  I_curves.tolist(),
            "final_mean":  soop2_I_m,
            "final_std":   soop2_I_s,
        },
        "sum_rate": {
            "mean_curve":  mean_R.tolist(),
            "std_curve":   std_R.tolist(),
            "all_curves":  R_curves.tolist(),
            "final_mean":  soop2_R_m,
            "final_std":   soop2_R_s,
        },
        "baselines": {
            "allon_I_mean": I_allon_m, "allon_I_std": I_allon_s,
            "rand_I_mean":  I_rand_m,  "rand_I_std":  I_rand_s,
            "allon_R_mean": R_allon_m, "allon_R_std": R_allon_s,
            "rand_R_mean":  R_rand_m,  "rand_R_std":  R_rand_s,
        },
        "elapsed_sec": elapsed,
    }
    with open(json_path, "w") as f:
        json.dump(_to_jsonable(pack), f, indent=2)
    print(f"[exp_SOOP2_testing] JSON  saved -> {json_path}")

    plt.show()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    main(n)
