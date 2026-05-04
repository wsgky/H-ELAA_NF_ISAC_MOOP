"""
Plotting helpers. Reads a JSON file produced by experiments/runner.py
and draws the corresponding figure with matplotlib.

Usage:
    python plot_results.py results/exp_power_xxx.json
    python plot_results.py results/exp_pareto_xxx.json
    python plot_results.py results/exp_convergence_xxx.json
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
    else:
        print("Unknown JSON schema."); return

    out_path = path.with_suffix(".png")
    fig.savefig(out_path, dpi=150)
    print(f"saved -> {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
