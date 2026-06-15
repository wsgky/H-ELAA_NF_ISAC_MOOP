"""
plot_v2.py -- figure generation for the Numerical-Results Simulation
Suite (v2).

Reads result dicts from ``result_v2/*.pkl`` (written by ``main_v2.py``)
and renders IEEE-Transactions-style figures (SciencePlots ``ieee``
style) into ``Figure/*.pdf`` and ``Figure/*.png``.

This module performs NO optimisation -- it only reads pickled result
dicts and calls matplotlib / numpy.

Usage
-----
    python plot_v2.py --fig all
    python plot_v2.py --fig pareto vs_power
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from experiments_v2 import _base


# ----------------------------------------------------------------------
# Module-level style configuration
# ----------------------------------------------------------------------
FONT_SIZE = 8
LABEL_SIZE = 8
TICK_SIZE = 7
LEGEND_SIZE = 7
FIG_WIDTH_IN = 3.5
FIG_HEIGHT_IN = 2.6
LINE_WIDTH = 1.0
MARKER_SIZE = 3.5
DPI_PNG = 600
FONT_FAMILY = "serif"

FIGURE_DIR = Path(__file__).resolve().parent / "Figure"


def _apply_style():
    """Apply SciencePlots IEEE style, then the module-level overrides above."""
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "ieee"])
    except Exception:
        pass

    # The "science"/"ieee" styles enable text.usetex, which requires a
    # LaTeX install. Fall back to mathtext (no external dependency) if
    # latex is not on PATH, so figures still render.
    if shutil.which("latex") is None:
        mpl.rcParams["text.usetex"] = False

    mpl.rcParams.update({
        "font.size": FONT_SIZE,
        "font.family": FONT_FAMILY,
        # "Times" (scienceplots default) is not installed on this
        # system; fall back to "Times New Roman" / DejaVu Serif so
        # serif rendering still works without findfont warnings.
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "lines.linewidth": LINE_WIDTH,
        "lines.markersize": MARKER_SIZE,
        "figure.figsize": (FIG_WIDTH_IN, FIG_HEIGHT_IN),
    })


def save_fig(fig, name: str):
    """Save ``fig`` as ``Figure/<name>.pdf`` and ``Figure/<name>.png``."""
    FIGURE_DIR.mkdir(exist_ok=True)
    pdf_path = FIGURE_DIR / f"{name}.pdf"
    png_path = FIGURE_DIR / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=DPI_PNG, bbox_inches="tight")
    plt.close(fig)
    print(f"[save_fig] '{name}' -> {pdf_path.name}, {png_path.name}")


# ----------------------------------------------------------------------
# Per-scheme style map (Sec. 5 benchmark schemes)
# ----------------------------------------------------------------------
SCHEME_STYLE = {
    "proposed":       dict(color="C0", marker="o", ls="-",
                            label="Proposed (Algorithm 1)"),
    "fully_digital":  dict(color="C1", marker="s", ls="--",
                            label="Fully-digital (upper bound)"),
    "amplitude_only": dict(color="C2", marker="^", ls="-.",
                            label="Amplitude-only RHS"),
    "random_rhs":     dict(color="C3", marker="v", ls=":",
                            label="Random RHS"),
    "uniform_rhs":    dict(color="C4", marker="D", ls="--",
                            label="Uniform RHS"),
    "far_field":      dict(color="C5", marker="x", ls="-.",
                            label="Far-field approximation"),
}


def _plot_scheme(ax, x, y, scheme, **kwargs):
    st = dict(SCHEME_STYLE.get(scheme, {}))
    label = st.pop("label", scheme)
    st.update(kwargs)
    ax.plot(x, y, label=label, markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH, **st)


def _load(name: str):
    if not _base.result_exists(name):
        print(f"[plot_v2] '{name}.pkl' not found in result_v2/ -- "
              f"run `python main_v2.py --exp {name.replace('exp_', '')}` first. Skipping.")
        return None
    return _base.load_result(name)


# ----------------------------------------------------------------------
# Fig. 3 -- convergence trajectories (tau, R, I vs outer iteration)
# ----------------------------------------------------------------------
def plot_convergence(name: str = "exp_convergence",
                      out_name: str = "fig3_convergence"):
    d = _load(name)
    if d is None:
        return

    x = np.asarray(d["x"])
    weights = d["weights"]
    tau, R, I = np.asarray(d["tau"]), np.asarray(d["R"]), np.asarray(d["I"])

    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_IN * 2.4, FIG_HEIGHT_IN))
    for wi, w1 in enumerate(weights):
        label = rf"$\omega_1={w1:.1f}$"
        axes[0].plot(x, tau[wi], marker="o", markersize=MARKER_SIZE,
                     linewidth=LINE_WIDTH, label=label)
        axes[1].plot(x, R[wi], marker="o", markersize=MARKER_SIZE,
                     linewidth=LINE_WIDTH, label=label)
        axes[2].plot(x, I[wi], marker="o", markersize=MARKER_SIZE,
                     linewidth=LINE_WIDTH, label=label)

    axes[0].set_ylabel(r"$\tau^{(s)}$")
    axes[1].set_ylabel(r"$R^{(s)}$ [bits/s/Hz]")
    axes[2].set_ylabel(r"$I^{(s)}$ [bits/s/Hz]")
    for ax in axes:
        ax.set_xlabel("Outer iteration $s$")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    save_fig(fig, out_name)


# ----------------------------------------------------------------------
# Fig. 4 -- CDF of outer iterations to converge
# ----------------------------------------------------------------------
def plot_iteration_cdf(name: str = "exp_iteration_cdf",
                        out_name: str = "fig4_iteration_cdf"):
    d = _load(name)
    if d is None:
        return

    weights = d["weights"]
    iter_counts = np.asarray(d["iter_counts"])  # (n_w, n_mc)
    Smax = d["meta"]["Smax"]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    for wi, w1 in enumerate(weights):
        counts = np.sort(iter_counts[wi])
        ys = np.arange(1, len(counts) + 1) / len(counts)
        ax.step(counts, ys, where="post", linewidth=LINE_WIDTH,
                label=rf"$\omega_1={w1:.1f}$")

    ax.set_xlabel("Outer iterations to converge")
    ax.set_ylabel("Empirical CDF")
    ax.set_xlim(1, Smax)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, out_name)


# ----------------------------------------------------------------------
# Pareto frontier (R vs I, sweeping omega1)
# ----------------------------------------------------------------------
def plot_pareto(name: str = "exp_pareto", out_name: str = "fig_pareto"):
    d = _load(name)
    if d is None:
        return

    schemes = d["schemes"]
    R, I = d["R"], d["I"]
    R_star = np.asarray(d["R_star"])
    I_star = np.asarray(d["I_star"])

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    for sc in schemes:
        order = np.argsort(R[sc])
        _plot_scheme(ax, np.asarray(R[sc])[order], np.asarray(I[sc])[order], sc)

    ax.scatter(R_star.mean(), I_star.mean(), marker="*", s=60,
                color="black", zorder=5, label="Ideal point $(R^\\star, I^\\star)$")

    ax.set_xlabel(r"Sum-rate $R$ [bits/s/Hz]")
    ax.set_ylabel(r"Sensing MI $I$ [bits/s/Hz]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, out_name)


# ----------------------------------------------------------------------
# Shared helper for vs_power / vs_mt / vs_k: 1x2 (R, I) vs x panel
# ----------------------------------------------------------------------
def _plot_vs_x(d, out_name, R_key="R", I_key="I", x_key="x",
               R_star_key="R_star", I_star_key="I_star",
               star_label=r"Single-objective optimum ($R^\star$/$I^\star$)"):
    schemes = d["schemes"]
    x = np.asarray(d[x_key])
    R, I = d[R_key], d[I_key]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN * 2.0, FIG_HEIGHT_IN + 0.7))
    for sc in schemes:
        _plot_scheme(axes[0], x, np.asarray(R[sc]), sc)
        _plot_scheme(axes[1], x, np.asarray(I[sc]), sc)

    star_kwargs = dict(color="black", linestyle=":", linewidth=LINE_WIDTH,
                        marker="*", markersize=MARKER_SIZE, label=star_label)
    if R_star_key in d:
        axes[0].plot(x, np.asarray(d[R_star_key]), **star_kwargs)
    if I_star_key in d:
        axes[1].plot(x, np.asarray(d[I_star_key]), **star_kwargs)

    axes[0].set_ylabel(r"Sum-rate $R$ [bits/s/Hz]")
    axes[1].set_ylabel(r"Sensing MI $I$ [bits/s/Hz]")
    for ax in axes:
        ax.set_xlabel(d["x_label"])
        ax.grid(True, alpha=0.3)

    # Both panels share the same set of lines/labels -- one figure-level
    # legend below the panels avoids the 7-8-entry legend box overflowing
    # a single ~3.5in-wide axes.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=LEGEND_SIZE,
               frameon=True)
    fig.tight_layout(rect=(0, 0.22, 1, 1))
    save_fig(fig, out_name)


# ----------------------------------------------------------------------
# R, I vs Pt
# ----------------------------------------------------------------------
def plot_vs_power(name: str = "exp_vs_power", out_name: str = "fig_vs_power"):
    d = _load(name)
    if d is None:
        return
    _plot_vs_x(d, out_name)


# ----------------------------------------------------------------------
# R, I vs Mt_h
# ----------------------------------------------------------------------
def plot_vs_mt(name: str = "exp_vs_mt", out_name: str = "fig_vs_mt"):
    d = _load(name)
    if d is None:
        return
    _plot_vs_x(d, out_name)


# ----------------------------------------------------------------------
# R, I vs Kc (Ks fixed) and vs Ks (Kc fixed)
# ----------------------------------------------------------------------
def plot_vs_k(name: str = "exp_vs_k", out_name: str = "fig_vs_k"):
    d = _load(name)
    if d is None:
        return

    kc_grid = d["meta"]["Kc_grid"]
    ks_fixed = d["meta"]["Ks_fixed_for_kc"]
    ks_grid = d["meta"]["Ks_grid"]
    kc_fixed = d["meta"]["Kc_fixed_for_ks"]

    d_kc = {
        "schemes": d["schemes"], "x": d["x_kc"],
        "x_label": rf"$K_c$ ($K_s={ks_fixed}$)",
        "R": d["R_kc"], "I": d["I_kc"],
        "R_star": d["R_star_kc"], "I_star": d["I_star_kc"],
    }
    d_ks = {
        "schemes": d["schemes"], "x": d["x_ks"],
        "x_label": rf"$K_s$ ($K_c={kc_fixed}$)",
        "R": d["R_ks"], "I": d["I_ks"],
        "R_star": d["R_star_ks"], "I_star": d["I_star_ks"],
    }

    _plot_vs_x(d_kc, f"{out_name}_Kc")
    _plot_vs_x(d_ks, f"{out_name}_Ks")

    print(f"  [plot_vs_k] Kc grid = {kc_grid}, Ks grid = {ks_grid}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
FIGURES = {
    "convergence": plot_convergence,
    "iteration_cdf": plot_iteration_cdf,
    "pareto": plot_pareto,
    "vs_power": plot_vs_power,
    "vs_mt": plot_vs_mt,
    "vs_k": plot_vs_k,
}


def main():
    parser = argparse.ArgumentParser(
        description="Render figures from result_v2/*.pkl into Figure/.")
    parser.add_argument(
        "--fig", nargs="+", default=["all"],
        choices=list(FIGURES.keys()) + ["all"],
        help="Which figure(s) to render (default: all).")
    args = parser.parse_args()

    _apply_style()
    FIGURE_DIR.mkdir(exist_ok=True)

    names = list(FIGURES.keys()) if "all" in args.fig else args.fig
    for name in names:
        print(f"--- [{name}] ---")
        FIGURES[name]()


if __name__ == "__main__":
    main()
