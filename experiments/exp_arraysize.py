"""
Experiment: performance vs number of TX RHS elements Mt (= Mt_h * Mt_v).

To keep the array roughly square we sweep Mt_h = Mt_v jointly via a
helper SystemConfig field; here we just sweep Mt_h and set Mt_v=Mt_h.
"""
import copy
from config import SystemConfig, AlgorithmConfig
from experiments.runner import run_sweep


def main():
    sys_cfg = SystemConfig(
        Mr_h=16, Mr_v=16, N=8,
        Kc=3, Ks=2, Ke=2, L=128, seed=2025,
        rhs_structure="subarray",
    )
    alg_cfg = AlgorithmConfig(outer_iters=8, inner_iters=15)

    Mh_list = [8, 12, 16, 20, 24]
    omega_list = [(0.5, 0.5)]

    # Use a custom sweep that ties Mt_v to Mt_h
    from experiments.runner import run_one_trial, _summarise, _to_jsonable, RESULTS_DIR
    import numpy as np, json, time

    all_results = []
    methods = ("SOOP1", "SOOP2", "MOOP")
    for Mh in Mh_list:
        per_param = []
        for trial in range(3):
            cfg = copy.deepcopy(sys_cfg)
            cfg.Mt_h = Mh
            cfg.Mt_v = Mh
            seed = sys_cfg.seed + 1000 * trial + 1
            res = run_one_trial(cfg, alg_cfg, seed,
                                methods=methods, omega_list=omega_list)
            per_param.append(res)
            print(f"   Mt_h=Mt_v={Mh} (Mt={Mh*Mh}), trial={trial+1}/3 done")
        all_results.append(per_param)

    summary = _summarise(all_results, methods, omega_list)
    pack = {"label": "exp_arraysize", "param_name": "Mt_h(=Mt_v)",
            "param_values": Mh_list,
            "n_trials": 3, "methods": list(methods),
            "omega_list": omega_list, "summary": summary}
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"exp_arraysize_{ts}.json"
    with open(path, "w") as f:
        json.dump(_to_jsonable(pack), f, indent=2)
    print(f"[exp_arraysize] saved -> {path}")


if __name__ == "__main__":
    main()
