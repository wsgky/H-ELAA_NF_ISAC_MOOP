"""
Plotting helpers. Reads a JSON file produced by experiments/runner.py
and draws the corresponding figure with matplotlib.

Usage:
    python plot_results.py results/exp_power_xxx.json
    python plot_results.py results/exp_pareto_xxx.json
    python plot_results.py results/exp_convergence_xxx.json
    python plot_results.py results/exp_montecarlo_xxx.json
"""
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_sweep(pack):
    pname = pack["param_name"]
    pvals = pack["param_values"]
    summary = pack["summary"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for m in ("SOOP1", "SOOP2"):
        if m in summary:
            axes[0].errorbar(pvals, summary[m]["sum_rate_mean"],
                             yerr=summary[m]["sum_rate_std"],
                             marker='o', label=m, capsize=3)
            axes[1].errorbar(pvals, summary[m]["sensing_mi_mean"],
                             yerr=summary[m]["sensing_mi_std"],
                             marker='o', label=m, capsize=3)

    if "MOOP" in summary:
        for entry in summary["MOOP"]:
            lbl = f"MOOP w=({entry['omega1']:.2f},{entry['omega2']:.2f})"
            axes[0].errorbar(pvals, entry["sum_rate_mean"],
                             yerr=entry["sum_rate_std"],
                             marker='s', linestyle='--',
                             label=lbl, capsize=3)
            axes[1].errorbar(pvals, entry["sensing_mi_mean"],
                             yerr=entry["sensing_mi_std"],
                             marker='s', linestyle='--',
                             label=lbl, capsize=3)

    axes[0].set_xlabel(pname); axes[0].set_ylabel("Sum-rate [bps/Hz]")
    axes[1].set_xlabel(pname); axes[1].set_ylabel("Sensing MI [bits]")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)
    fig.suptitle(pack.get("label", "sweep"))
    fig.tight_layout()
    return fig


def plot_pareto(pack):
    R = np.array(pack["R_mean"])
    I = np.array(pack["I_mean"])
    omegas = pack["omegas"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(R, I, '-o', color='C0')
    for (w1, w2), r, mi in zip(omegas, R, I):
        ax.annotate(f"({w1:.2f},{w2:.2f})", xy=(r, mi), fontsize=7,
                    xytext=(3, 3), textcoords='offset points')
    ax.set_xlabel("Sum-rate [bps/Hz]")
    ax.set_ylabel("Sensing MI [bits]")
    ax.set_title("Pareto frontier (MOOP, sweeping omega1)")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    return fig


def plot_convergence(pack):
    structures = pack["structures"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for struct, data in structures.items():
        axes[0].plot(data["SOOP1_history"]["sum_rate"],
                     marker='o', label=f"SOOP1 [{struct}]")
        axes[0].plot(data["MOOP_history"]["sum_rate"],
                     marker='s', label=f"MOOP [{struct}]")
        axes[1].plot(data["SOOP2_history"]["sensing_mi"],
                     marker='o', label=f"SOOP2 [{struct}]")
        axes[1].plot(data["MOOP_history"]["sensing_mi"],
                     marker='s', label=f"MOOP [{struct}]")
    axes[0].set_xlabel("outer iter"); axes[0].set_ylabel("Sum-rate [bps/Hz]")
    axes[1].set_xlabel("outer iter"); axes[1].set_ylabel("Sensing MI [bits]")
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(True, alpha=0.4)
    fig.suptitle("Convergence: sub-array vs fully-connected RHS")
    fig.tight_layout()
    return fig


def plot_montecarlo(pack):
    """Four-panel figure for the Monte Carlo validation experiment."""
    summary = pack["summary"]
    n_trials = pack["n_trials"]

    # ── collect per-trial arrays ──────────────────────────────────────────
    R1 = np.array(summary["SOOP1"]["sum_rate_all"])
    I1 = np.array(summary["SOOP1"]["sensing_mi_all"])
    R2 = np.array(summary["SOOP2"]["sum_rate_all"])
    I2 = np.array(summary["SOOP2"]["sensing_mi_all"])

    # keep only MOOP entries that have per-trial arrays
    moop_entries = [e for e in summary["MOOP"] if "sum_rate_all" in e]

    # ── colour / marker scheme ────────────────────────────────────────────
    colours = {"SOOP1": "C0", "SOOP2": "C1"}
    moop_colours = ["C2", "C3", "C4"]

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32)
    ax_box_r  = fig.add_subplot(gs[0, 0])   # box plot: sum-rate
    ax_box_i  = fig.add_subplot(gs[0, 1])   # box plot: sensing MI
    ax_cdf_r  = fig.add_subplot(gs[1, 0])   # CDF: sum-rate
    ax_cdf_ri = fig.add_subplot(gs[1, 1])   # R-I mean ± std scatter

    # ── [0,0] Box plots — sum-rate ────────────────────────────────────────
    box_data_r  = [R1, R2] + [np.array(e["sum_rate_all"])   for e in moop_entries]
    box_labels_r = ["SOOP1", "SOOP2"] + [
        f"MOOP\n({e['omega1']:.1f}/{e['omega2']:.1f})" for e in moop_entries]
    bp = ax_box_r.boxplot(box_data_r, tick_labels=box_labels_r,
                          patch_artist=True, widths=0.45,
                          medianprops=dict(color="black", linewidth=1.5))
    box_colours = [colours["SOOP1"], colours["SOOP2"]] + moop_colours[:len(moop_entries)]
    for patch, c in zip(bp["boxes"], box_colours):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    ax_box_r.set_ylabel("Sum-rate [bits/Hz]")
    ax_box_r.set_title(f"Sum-rate distribution ({n_trials} trials)")
    ax_box_r.grid(True, axis='y', alpha=0.4)

    # ── [0,1] Box plots — sensing MI ─────────────────────────────────────
    box_data_i   = [I1, I2] + [np.array(e["sensing_mi_all"]) for e in moop_entries]
    bp2 = ax_box_i.boxplot(box_data_i, tick_labels=box_labels_r,
                           patch_artist=True, widths=0.45,
                           medianprops=dict(color="black", linewidth=1.5))
    for patch, c in zip(bp2["boxes"], box_colours):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    ax_box_i.set_ylabel("Sensing MI [bits/Hz]")
    ax_box_i.set_title(f"Sensing MI distribution ({n_trials} trials)")
    ax_box_i.grid(True, axis='y', alpha=0.4)

    # ── [1,0] CDF — sum-rate ──────────────────────────────────────────────
    def _ecdf(x):
        xs = np.sort(x)
        return xs, np.arange(1, len(xs) + 1) / len(xs)

    for label, data, c in [("SOOP1", R1, colours["SOOP1"]),
                            ("SOOP2", R2, colours["SOOP2"])]:
        xs, ys = _ecdf(data)
        ax_cdf_r.step(xs, ys, where='post', label=label, color=c, linewidth=1.8)

    for e, c in zip(moop_entries, moop_colours):
        xs, ys = _ecdf(np.array(e["sum_rate_all"]))
        lbl = f"MOOP({e['omega1']:.1f}/{e['omega2']:.1f})"
        ax_cdf_r.step(xs, ys, where='post', label=lbl, color=c,
                      linestyle='--', linewidth=1.8)

    ax_cdf_r.set_xlabel("Sum-rate [bits/Hz]")
    ax_cdf_r.set_ylabel("Empirical CDF")
    ax_cdf_r.set_title("CDF of sum-rate across Monte Carlo trials")
    ax_cdf_r.legend(fontsize=8)
    ax_cdf_r.grid(True, alpha=0.4)

    # ── [1,1] R-I scatter: mean ± std error bars ─────────────────────────
    def _plot_point(ax, r_mean, r_std, i_mean, i_std, label, color, marker):
        ax.errorbar(r_mean, i_mean,
                    xerr=r_std, yerr=i_std,
                    fmt=marker, color=color, capsize=5,
                    markersize=8, linewidth=1.5, label=label)

    _plot_point(ax_cdf_ri,
                summary["SOOP1"]["sum_rate_mean"], summary["SOOP1"]["sum_rate_std"],
                summary["SOOP1"]["sensing_mi_mean"], summary["SOOP1"]["sensing_mi_std"],
                "SOOP1", colours["SOOP1"], "o")
    _plot_point(ax_cdf_ri,
                summary["SOOP2"]["sum_rate_mean"], summary["SOOP2"]["sum_rate_std"],
                summary["SOOP2"]["sensing_mi_mean"], summary["SOOP2"]["sensing_mi_std"],
                "SOOP2", colours["SOOP2"], "s")

    for e, c in zip(moop_entries, moop_colours):
        _plot_point(ax_cdf_ri,
                    e["sum_rate_mean"], e["sum_rate_std"],
                    e["sensing_mi_mean"], e["sensing_mi_std"],
                    f"MOOP({e['omega1']:.1f}/{e['omega2']:.1f})", c, "^")

    ax_cdf_ri.set_xlabel("Sum-rate [bits/Hz]")
    ax_cdf_ri.set_ylabel("Sensing MI [bits/Hz]")
    ax_cdf_ri.set_title("R-I tradeoff: mean ± std over Monte Carlo")
    ax_cdf_ri.legend(fontsize=8)
    ax_cdf_ri.grid(True, alpha=0.4)

    cfg = pack.get("sys_cfg", {})
    title = (f"Monte Carlo Validation — {n_trials} trials | "
             f"Mt={cfg.get('Mt_h','')}×{cfg.get('Mt_v','')} "
             f"N={cfg.get('N','')} "
             f"Kc={cfg.get('Kc','')} Ks={cfg.get('Ks','')} "
             f"Pt={cfg.get('Pt_dBm','')} dBm | "
             f"{cfg.get('rhs_structure','')}")
    fig.suptitle(title, fontsize=10)
    return fig


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    path = Path(sys.argv[1])
    with open(path) as f:
        pack = json.load(f)

    if "param_name" in pack:
        fig = plot_sweep(pack)
    elif "omegas" in pack:
        fig = plot_pareto(pack)
    elif "structures" in pack:
        fig = plot_convergence(pack)
    elif pack.get("label") == "exp_montecarlo":
        fig = plot_montecarlo(pack)
    else:
        print("Unknown JSON schema."); return

    out_path = path.with_suffix(".png")
    fig.savefig(out_path, dpi=150)
    print(f"saved -> {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
