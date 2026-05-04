"""
Experiment runner: loop over Monte-Carlo trials and parameter sweeps,
save results to .npz / .json so they can be re-plotted later without
re-running the simulation.
"""
import os
import json
import time
import copy
from pathlib import Path
import numpy as np

from config import SystemConfig, AlgorithmConfig
from system_model import generate_scenario
from algorithms import solve_SOOP1, solve_SOOP2, solve_MOOP


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_one_trial(sys_cfg: SystemConfig, alg_cfg: AlgorithmConfig,
                  seed: int, methods=("SOOP1", "SOOP2", "MOOP"),
                  omega_list=None) -> dict:
    """
    Run one Monte-Carlo trial: generate scenario then run the requested
    methods.
    """
    sys_cfg = copy.deepcopy(sys_cfg)
    sys_cfg.seed = seed
    rng = np.random.default_rng(seed)
    scenario = generate_scenario(sys_cfg, rng=rng)

    out = {"seed": seed}

    soop1_res = soop2_res = None
    if "SOOP1" in methods or "MOOP" in methods:
        soop1_res = solve_SOOP1(scenario, sys_cfg, alg_cfg)
        out["SOOP1"] = {
            "sum_rate": soop1_res["history"]["sum_rate"][-1],
            "sensing_mi": soop1_res["history"]["sensing_mi"][-1],
            "history": soop1_res["history"],
        }
    if "SOOP2" in methods or "MOOP" in methods:
        soop2_res = solve_SOOP2(scenario, sys_cfg, alg_cfg)
        out["SOOP2"] = {
            "sum_rate": soop2_res["history"]["sum_rate"][-1],
            "sensing_mi": soop2_res["history"]["sensing_mi"][-1],
            "history": soop2_res["history"],
        }

    if "MOOP" in methods:
        if omega_list is None:
            omega_list = [(alg_cfg.omega1, alg_cfg.omega2)]
        out["MOOP"] = []
        for w1, w2 in omega_list:
            ac = copy.deepcopy(alg_cfg)
            ac.omega1, ac.omega2 = w1, w2
            res = solve_MOOP(scenario, sys_cfg, ac,
                             omega1=w1, omega2=w2,
                             soop1_result=soop1_res,
                             soop2_result=soop2_res)
            out["MOOP"].append({
                "omega1": w1, "omega2": w2,
                "sum_rate": res["history"]["sum_rate"][-1],
                "sensing_mi": res["history"]["sensing_mi"][-1],
                "history": res["history"],
            })

    return out


def run_sweep(label: str,
              sys_cfg: SystemConfig, alg_cfg: AlgorithmConfig,
              param_name: str, param_values: list,
              n_trials: int = 5,
              methods=("SOOP1", "SOOP2", "MOOP"),
              omega_list=None,
              save: bool = True) -> dict:
    """
    Sweep `param_name` (a SystemConfig field) over `param_values`,
    averaging across `n_trials` Monte-Carlo realisations.

    Returns
    -------
    dict with keys
        param_name, param_values, methods, results (n_param x n_trials)
    """
    print(f"[run_sweep] {label} | sweep {param_name} over {param_values}")
    all_results = []
    t0 = time.time()
    for p in param_values:
        per_param = []
        for trial in range(n_trials):
            cfg = copy.deepcopy(sys_cfg)
            setattr(cfg, param_name, p)
            seed = sys_cfg.seed + 1000 * trial + 1
            res = run_one_trial(cfg, alg_cfg, seed,
                                methods=methods, omega_list=omega_list)
            per_param.append(res)
            print(f"   {param_name}={p}, trial={trial+1}/{n_trials} done")
        all_results.append(per_param)

    summary = _summarise(all_results, methods, omega_list)
    pack = {
        "label": label,
        "param_name": param_name,
        "param_values": list(param_values),
        "n_trials": n_trials,
        "methods": list(methods),
        "omega_list": omega_list,
        "summary": summary,
        "elapsed_sec": time.time() - t0,
    }

    if save:
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = RESULTS_DIR / f"{label}_{param_name}_{ts}.json"
        with open(path, "w") as f:
            json.dump(_to_jsonable(pack), f, indent=2)
        print(f"[run_sweep] saved -> {path}")

    return pack


def _summarise(all_results, methods, omega_list):
    """Average sum_rate / sensing_mi across trials, per parameter value."""
    summary = {m: {"sum_rate_mean": [], "sum_rate_std": [],
                    "sensing_mi_mean": [], "sensing_mi_std": []}
               for m in methods if m in ("SOOP1", "SOOP2")}
    if "MOOP" in methods:
        n_omegas = len(omega_list) if omega_list else 1
        summary["MOOP"] = []
        for o in range(n_omegas):
            summary["MOOP"].append(
                {"omega1": omega_list[o][0] if omega_list else None,
                 "omega2": omega_list[o][1] if omega_list else None,
                 "sum_rate_mean": [], "sum_rate_std": [],
                 "sensing_mi_mean": [], "sensing_mi_std": []})

    for per_param in all_results:
        for m in ("SOOP1", "SOOP2"):
            if m not in methods:
                continue
            R = [r[m]["sum_rate"] for r in per_param]
            I = [r[m]["sensing_mi"] for r in per_param]
            summary[m]["sum_rate_mean"].append(float(np.mean(R)))
            summary[m]["sum_rate_std"].append(float(np.std(R)))
            summary[m]["sensing_mi_mean"].append(float(np.mean(I)))
            summary[m]["sensing_mi_std"].append(float(np.std(I)))
        if "MOOP" in methods:
            for o, _ in enumerate(omega_list or [None]):
                R = [r["MOOP"][o]["sum_rate"] for r in per_param]
                I = [r["MOOP"][o]["sensing_mi"] for r in per_param]
                summary["MOOP"][o]["sum_rate_mean"].append(float(np.mean(R)))
                summary["MOOP"][o]["sum_rate_std"].append(float(np.std(R)))
                summary["MOOP"][o]["sensing_mi_mean"].append(float(np.mean(I)))
                summary["MOOP"][o]["sensing_mi_std"].append(float(np.std(I)))

    return summary


def _to_jsonable(obj):
    """Recursively convert numpy arrays etc. to JSON-friendly types."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, complex):
        return {"re": obj.real, "im": obj.imag}
    return obj
