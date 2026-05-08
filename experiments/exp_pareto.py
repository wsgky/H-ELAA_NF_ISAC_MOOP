"""
Experiment: Pareto frontier of MOOP, by sweeping (omega1, omega2) along
the simplex omega1 + omega2 = 1.
"""
import copy, json, time, numpy as np
from config import SystemConfig, AlgorithmConfig
from experiments.runner import run_one_trial, _to_jsonable, RESULTS_DIR


def main():
    sys_cfg = SystemConfig(
        Mt_h=16, Mt_v=16, Mr_h=16, Mr_v=16, N=8,
        Kc=3, Ks=2, Ke=2, L=128, seed=2025,
        rhs_structure="subarray",
    )
    alg_cfg = AlgorithmConfig(outer_iters=8, inner_iters=15)

    omegas = [(w, 1.0 - w) for w in np.linspace(0.05, 0.95, 9)]
    n_trials = 3

    # We collect (R, I) pairs per trial and average across trials.
    R_arr = np.zeros((len(omegas), n_trials))
    I_arr = np.zeros((len(omegas), n_trials))
    for trial in range(n_trials):
        cfg = copy.deepcopy(sys_cfg)
        seed = sys_cfg.seed + 1000 * trial + 1
        # one full call per trial covers all omegas (re-uses R*, I*)
        res = run_one_trial(cfg, alg_cfg, seed,
                            methods=("SOOP1", "SOOP2", "MOOP"),
                            omega_list=omegas)
        for o, _ in enumerate(omegas):
            R_arr[o, trial] = res["MOOP"][o]["sum_rate"]
            I_arr[o, trial] = res["MOOP"][o]["sensing_mi"]
        print(f"   trial {trial+1}/{n_trials} done")

    pack = {
        "label": "exp_pareto",
        "omegas": omegas,
        "R_mean": R_arr.mean(1).tolist(),
        "R_std": R_arr.std(1).tolist(),
        "I_mean": I_arr.mean(1).tolist(),
        "I_std": I_arr.std(1).tolist(),
    }
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"exp_pareto_{ts}.json"
    with open(path, "w") as f:
        json.dump(_to_jsonable(pack), f, indent=2)
    print(f"[exp_pareto] saved -> {path}")


if __name__ == "__main__":
    main()
