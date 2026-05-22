"""
Plot results for exp_power_sensing and exp_ks_sensing.

Automatically loads the most recent JSON for each experiment from results/.
Generates two separate figures (each saved as PNG + PDF):
  Figure 1 — Sensing MI vs transmit power (Pt_dBm)
  Figure 2 — Sensing MI vs number of sensing targets (Ks)

Both RHS structures (fully_connected / subarray) are shown on the same axes:
  solid line  = fully_connected
  dashed line = subarray

Only Monte Carlo mean values are plotted (no variance bands).

Usage:
    python experiments/plot_sensing_sweeps.py
    python experiments/plot_sensing_sweeps.py <power_json> <ks_json>
"""
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── SciencePlots style ─────────────────────────────────────────────────────────
try:
    import scienceplots  # registers styles as a side-effect
    plt.style.use(["science", "no-latex", "grid"])
except ImportError:
    plt.style.use("seaborn-v0_8-whitegrid")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# ── visual encoding ────────────────────────────────────────────────────────────
# Each method: same color for FC (solid) and Sub (dashed); distinct marker pair.
METHODS = {
    "soop2":  dict(color="#19A3D9", marker_fc="o",  marker_sub="o",  label="Proposed"),
    "noholo": dict(color="#FF0022", marker_fc="s",  marker_sub="s",  label="Fully digital"),
    "allon":  dict(color="#698000", marker_fc="^",  marker_sub="^",  label="All-on ($a$=1)"),
    "rand":   dict(color="#8E2F89", marker_fc="D",  marker_sub="D",  label="Random"),
}
METHODS2 = {
    "soop2":  dict(color="#19A3D9", marker_fc="o",  marker_sub="o",  label="Proposed"),
    # "noholo": dict(color="C1", marker_fc="s",  marker_sub="s",  label="Fully digital"),
    # "allon":  dict(color="C2", marker_fc="^",  marker_sub="^",  label="All-on ($a$=1)"),
    # "rand":   dict(color="C3", marker_fc="D",  marker_sub="D",  label="Random"),
}
LS_FC  = "-"
LS_SUB = "--"


def _load_latest(prefix: str) -> dict:
    files = sorted(RESULTS_DIR.glob(f"{prefix}_*.json"))
    if not files:
        raise FileNotFoundError(
            f"No JSON found for prefix '{prefix}' in {RESULTS_DIR}\n"
            f"Run the corresponding experiment first.")
    path = files[-1]
    print(f"  Loading {path.name}")
    with open(path) as f:
        return json.load(f)


def _plot_sweep(data: dict, x_key: str, xlabel: str, title: str):
    """
    Single figure: FC (solid) and Sub (dashed) on the same axes.
    Only mean values plotted.
    """
    x   = np.array(data[x_key])
    fig, ax = plt.subplots(figsize=(4.5, 4))

    for m, s in METHODS.items():
        # ---- fully connected ----
        mean_fc = np.array(data["fully_connected"][m]["mean"], dtype=float)
        mask_fc = ~np.isnan(mean_fc)
        ax.plot(x[mask_fc], mean_fc[mask_fc],
                color=s["color"], linestyle=LS_FC,
                marker=s["marker_fc"], markersize=6, linewidth=1.5,
                fillstyle="full",
                label=f"{s['label']}")
        
    # for m, s in METHODS2.items():
    #     # ---- subarray ----
    #     mean_sub = np.array(data["subarray"][m]["mean"], dtype=float)
    #     mask_sub = ~np.isnan(mean_sub)
    #     if mask_sub.any():
    #         ax.plot(x[mask_sub], mean_sub[mask_sub],
    #                 color=s["color"], linestyle=LS_SUB,
    #                 marker=s["marker_sub"], markersize=6, linewidth=1.5,
    #                 fillstyle="none",
    #                 label=f"{s['label']}, Sub")

    ax.set_xlabel(xlabel,fontsize=18)
    ax.set_ylabel("Sensing MI [bits/Hz]",fontsize=18)
    # ax.set_title(title,fontsize=14)
    # # ax.set_xlim(left=4.6,right=30.4)
    # # ax.set_ylim(bottom=10,top=60)
    # ax.set_xlim(left=1.8,right=10.2)
    # ax.set_ylim(bottom=0,top=250)

    # ax.set_xlim(left=5,right=30)
    # ax.set_ylim(bottom=10,top=60)
    ax.set_xlim(left=2,right=10)
    ax.set_ylim(bottom=0,top=250)

    ax.set_xticks(x)
    # ax.legend(fontsize=12, ncol=1, loc="upper left", frameon=True)
    ax.legend(fontsize=12, ncol=1, loc="lower right", frameon=True)
    fig.tight_layout()
    return fig


def _save(fig, stem: str):
    ts   = time.strftime("%Y%m%d_%H%M%S")
    png  = RESULTS_DIR / f"{stem}_{ts}.png"
    pdf  = RESULTS_DIR / f"{stem}_{ts}.pdf"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(pdf,           bbox_inches="tight")
    print(f"  PNG -> {png}")
    print(f"  PDF -> {pdf}")


def main():
    if len(sys.argv) == 3:
        with open(sys.argv[1]) as f: power_data = json.load(f)
        with open(sys.argv[2]) as f: ks_data    = json.load(f)
    else:
        print("Auto-loading latest results...")
        power_data = _load_latest("exp_power_sensing")
        ks_data    = _load_latest("exp_ks_sensing")

    fig1 = _plot_sweep(
        power_data,
        x_key  = "Pt_dBm",
        xlabel = "$P_t$ [dBm]",
        title  = "Sensing MI vs Transmit Power",
    )
    _save(fig1, "plot_power_sensing")

    fig2 = _plot_sweep(
        ks_data,
        x_key  = "Ks",
        xlabel = "Number of Sensing Targets $K_s$",
        title  = "Sensing MI vs Number of Sensing Targets",
    )
    _save(fig2, "plot_ks_sensing")

    plt.show()


if __name__ == "__main__":
    main()
