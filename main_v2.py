"""
main_v2.py -- CLI driver for the Numerical-Results Simulation Suite (v2).

Runs the experiments in ``experiments_v2/exp_*.py`` and saves their
result dicts to ``result_v2/`` via ``experiments_v2._base.save_result``.

This module performs NO plotting (no matplotlib import) -- see
``plot_v2.py`` for figure generation, which reads exclusively from
``result_v2/``.

Usage
-----
    python main_v2.py --exp smoke_test          # (not a real exp; see --smoke)
    python main_v2.py --exp convergence pareto
    python main_v2.py --exp all
    python main_v2.py --smoke --exp all          # fast sanity run
    python main_v2.py --exp all --overwrite      # force re-run + overwrite
"""
import argparse
import time

import config_v2
from experiments_v2 import _base
from experiments_v2 import (
    exp_convergence,
    exp_iteration_cdf,
    exp_pareto,
    exp_vs_power,
    exp_vs_mt,
    exp_vs_k,
)


EXPERIMENTS = {
    "convergence": exp_convergence,
    "iteration_cdf": exp_iteration_cdf,
    "pareto": exp_pareto,
    "vs_power": exp_vs_power,
    "vs_mt": exp_vs_mt,
    "vs_k": exp_vs_k,
}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run experiments_v2 numerical-results simulations.")
    parser.add_argument(
        "--exp", nargs="+", default=["all"],
        choices=list(EXPERIMENTS.keys()) + ["all"],
        help="Which experiment(s) to run (default: all).")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Use config_v2.get_config(smoke=True): small array, "
             "n_mc=3, coarse sweeps -- fast sanity run.")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-run and overwrite results that already exist in "
             "result_v2/ (default: skip experiments whose .pkl already "
             "exists).")
    return parser.parse_args()


def _resolve_names(requested):
    if "all" in requested:
        return list(EXPERIMENTS.keys())
    # de-duplicate while preserving order
    seen = set()
    names = []
    for name in requested:
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def main():
    args = _parse_args()
    names = _resolve_names(args.exp)
    cfg = config_v2.get_config(smoke=args.smoke)

    print("=" * 70)
    print(f"main_v2: running {names}  (smoke={args.smoke}, "
          f"overwrite={args.overwrite})")
    print(f"  n_mc={cfg.n_mc}  Smax={cfg.Smax}  schemes={cfg.scheme_list()}")
    print("=" * 70)

    for name in names:
        out_name = f"exp_{name}"
        if _base.result_exists(out_name) and not args.overwrite:
            print(f"[{name}] result_v2/{out_name}.pkl already exists "
                  f"-- skipping (use --overwrite to re-run)")
            continue

        print(f"\n--- [{name}] starting ---")
        t0 = time.time()
        result = EXPERIMENTS[name].run(cfg)
        dt = time.time() - t0
        print(f"--- [{name}] done in {dt:.1f}s ---")

        _base.save_result(out_name, result)

    print("\nmain_v2: all requested experiments finished.")


if __name__ == "__main__":
    main()
