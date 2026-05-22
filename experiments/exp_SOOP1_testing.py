"""
SOOP1 fine-grained convergence validation via Monte Carlo.

For every trial the convergence curve is unrolled as:
    outer iter 0 : [R_after_WF, R_inner_1, ..., R_inner_T]
    outer iter 1 : [R_after_WF, R_inner_1, ..., R_inner_T]
    ...
giving  outer_iters x (1 + inner_iters)  points per trial.

Usage:
    python main.py SOOP1_testing            # default 20 trials
    python main.py SOOP1_testing 50
    python experiments/exp_SOOP1_testing.py 20
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
from algorithms import solve_SOOP1
from algorithms.soop1_comm import _zf_waterfilling
from algorithms.utils import compute_F, sum_rate


def _wf_only_R(scen, cfg, a: np.ndarray) -> float:
    """ZF + water-filling with fixed a. Returns sum-rate."""
    W_s = np.zeros((scen.N, scen.Ks), dtype=complex)
    F   = compute_F(a, scen.Phi)
    W_c, _ = _zf_waterfilling(scen.H, F, cfg.sigma2, cfg.Pt)
    W = np.concatenate([W_c, W_s], axis=1)
    return float(sum_rate(scen.H, F, W, cfg.sigma2, scen.Kc))


def main(n_trials: int = 20):
    sys_cfg = SystemConfig(
        Mt_h=128, Mt_v=8, Mr_h=128, Mr_v=8, N=8,
        Kc=3, Ks=2, Ke=2, L=128, seed=2025,
        rhs_structure="fully_connected",
    )
    alg_cfg = AlgorithmConfig(outer_iters=1, inner_iters=100000)

    OI = alg_cfg.outer_iters
    # each outer iter: 1 WF point + inner_iters PGD points
    steps_per_outer = 1 + alg_cfg.inner_iters
    total_steps     = OI * steps_per_outer

    print(f"[exp_SOOP1_testing] {n_trials} trials | "
          f"Mt={sys_cfg.Mt_h}x{sys_cfg.Mt_v}  N={sys_cfg.N} "
          f"Kc={sys_cfg.Kc}  Ks={sys_cfg.Ks}  Ke={sys_cfg.Ke} "
          f"Pt={sys_cfg.Pt_dBm} dBm")
    print(f"  outer_iters={OI}  inner_iters={alg_cfg.inner_iters}"
          f"  pgd_steps={alg_cfg.pgd_steps}"
          f"  => {total_steps} points/trial")
    print("-" * 65)

    all_curves   = []
    final_R_list = []
    R_allon_list = []
    R_rand_list  = []

    t0 = time.time()

    for trial in range(n_trials):
        seed = sys_cfg.seed + trial + 1
        cfg  = copy.deepcopy(sys_cfg)
        cfg.seed = seed
        rng  = np.random.default_rng(seed)
        scen = generate_scenario(cfg, rng)

        # ── baselines ─────────────────────────────────────────────────────
        R_allon = _wf_only_R(scen, cfg, np.ones(scen.Mt))
        R_allon_list.append(R_allon)
        a_rand = np.random.default_rng(seed + 500).uniform(0, 1, scen.Mt)
        R_rand = _wf_only_R(scen, cfg, a_rand)
        R_rand_list.append(R_rand)

        # ── SOOP1 with fine-grained history ───────────────────────────────
        res = solve_SOOP1(scen, cfg, alg_cfg, full_history=True)
        h   = res["history"]

        # unroll: [R_wf, inner_1, ..., inner_T] per outer iter
        curve = []
        for seg in h["inner_sum_rate"]:
            curve.extend(seg)
        if len(curve) < total_steps:
            curve += [curve[-1]] * (total_steps - len(curve))
        all_curves.append(curve[:total_steps])

        final_R = h["sum_rate"][-1]
        final_R_list.append(final_R)

        mono_ok = all(b >= a - 1e-6
                      for a, b in zip(h["sum_rate"], h["sum_rate"][1:]))
        flag = "" if mono_ok else "  [MONO VIOLATION]"
        print(f"  trial {trial+1:3d}/{n_trials} | "
              f"R={final_R:.3f}  R_allon={R_allon:.3f}  R_rand={R_rand:.3f}{flag}")

    elapsed = time.time() - t0

    curves = np.array(all_curves)      # (n_trials, total_steps)
    mean_c = curves.mean(axis=0)
    std_c  = curves.std(axis=0)
    x      = np.arange(total_steps)

    R_allon_arr = np.array(R_allon_list)
    R_rand_arr  = np.array(R_rand_list)
    allon_m, allon_s = float(R_allon_arr.mean()), float(R_allon_arr.std())
    rand_m,  rand_s  = float(R_rand_arr.mean()),  float(R_rand_arr.std())
    soop1_m = float(np.mean(final_R_list))
    soop1_s = float(np.std(final_R_list))

    # ── summary ────────────────────────────────────────────────────────────
    W = 65
    print("\n" + "=" * W)
    print(f"  {'Algorithm':<22}  {'Mean R':>8}  {'Std R':>7}  {'vs rand-A':>10}")
    print("  " + "-" * (W - 2))
    print(f"  {'SOOP1 (final)':<22}  {soop1_m:>8.4f}  {soop1_s:>7.4f}"
          f"  {soop1_m - rand_m:>+10.4f}")
    print(f"  {'All-on (a=1)':<22}  {allon_m:>8.4f}  {allon_s:>7.4f}"
          f"  {allon_m - rand_m:>+10.4f}")
    print(f"  {'Random-A':<22}  {rand_m:>8.4f}  {rand_s:>7.4f}"
          f"  {'(reference)':>10}")
    print("=" * W)
    print(f"\n  Elapsed : {elapsed:.1f} s\n")

    # ── plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))

    for curve in all_curves:
        ax.plot(x, curve, color='C0', alpha=0.12, linewidth=0.7)
    ax.fill_between(x, mean_c - std_c, mean_c + std_c,
                    alpha=0.30, color='C0', label='±1 std')
    ax.plot(x, mean_c, color='C0', linewidth=2.0, label='Mean')

    ax.axhline(allon_m, color='C2', linestyle='-.', linewidth=1.5,
               label=f'All-on mean ({allon_m:.2f})')
    ax.fill_between(x, [allon_m - allon_s] * total_steps,
                       [allon_m + allon_s] * total_steps,
                    alpha=0.10, color='C2')
    ax.axhline(rand_m, color='C3', linestyle=':', linewidth=1.5,
               label=f'Random-A mean ({rand_m:.2f})')
    ax.fill_between(x, [rand_m - rand_s] * total_steps,
                       [rand_m + rand_s] * total_steps,
                    alpha=0.10, color='C3')

    # outer-iteration boundary markers
    for it in range(OI + 1):
        ax.axvline(it * steps_per_outer, color='gray',
                   linestyle='--', linewidth=0.7, alpha=0.5)

    # top x-axis: outer iteration labels
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ticks = [it * steps_per_outer + steps_per_outer // 2 for it in range(OI)]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"Out {it}" for it in range(OI)], fontsize=7)
    ax2.tick_params(length=0)

    ax.set_xlabel(
        f"Cumulative step  (1 WF + {alg_cfg.inner_iters} inner PGD  per outer iter)")
    ax.set_ylabel("Sum-rate [bits/Hz]")
    ax.set_title("Fine-grained convergence  (SOOP1)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)

    cfg_str = (f"Mt={sys_cfg.Mt_h}x{sys_cfg.Mt_v}  N={sys_cfg.N}  "
               f"Kc={sys_cfg.Kc}  Ks={sys_cfg.Ks}  "
               f"Pt={sys_cfg.Pt_dBm} dBm  {sys_cfg.rhs_structure}")
    fig.suptitle(f"SOOP1 Monte Carlo — {n_trials} trials\n{cfg_str}", fontsize=10)
    fig.tight_layout()

    ts = time.strftime("%Y%m%d_%H%M%S")
    png_path = RESULTS_DIR / f"exp_soop1_convergence_{ts}.png"
    fig.savefig(png_path, dpi=150)
    print(f"[exp_SOOP1_testing] figure saved -> {png_path}")

    json_path = RESULTS_DIR / f"exp_soop1_convergence_{ts}.json"
    pack = {
        "label":         "exp_soop1_convergence",
        "n_trials":      n_trials,
        "outer_iters":   OI,
        "inner_iters":   alg_cfg.inner_iters,
        "pgd_steps":     alg_cfg.pgd_steps,
        "steps_per_outer": steps_per_outer,
        "sys_cfg": {
            "Mt_h": sys_cfg.Mt_h, "Mt_v": sys_cfg.Mt_v,
            "N": sys_cfg.N, "Kc": sys_cfg.Kc,
            "Ks": sys_cfg.Ks, "Ke": sys_cfg.Ke,
            "L": sys_cfg.L, "Pt_dBm": sys_cfg.Pt_dBm,
            "rhs_structure": sys_cfg.rhs_structure,
        },
        "mean_curve":   mean_c.tolist(),
        "std_curve":    std_c.tolist(),
        "all_curves":   curves.tolist(),
        "final_R_mean": soop1_m,
        "final_R_std":  soop1_s,
        "baselines": {
            "allon_mean": allon_m, "allon_std": allon_s,
            "rand_mean":  rand_m,  "rand_std":  rand_s,
        },
        "elapsed_sec": elapsed,
    }
    with open(json_path, "w") as f:
        json.dump(_to_jsonable(pack), f, indent=2)
    print(f"[exp_SOOP1_testing] JSON  saved -> {json_path}")

    plt.show()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    main(n)
