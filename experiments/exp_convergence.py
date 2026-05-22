"""
Experiment: convergence curves of all three algorithms, and structure
comparison (fully-connected vs sub-array RHS).
"""
import copy, json, time, numpy as np
from config import SystemConfig, AlgorithmConfig
from experiments.runner import RESULTS_DIR, _to_jsonable
from system_model import generate_scenario
from algorithms import solve_SOOP1, solve_SOOP2, solve_MOOP


def main():
    sys_cfg_base = SystemConfig(
        Mt_h=128, Mt_v=8, Mr_h=128, Mr_v=8, N=8,
        Kc=3, Ks=2, Ke=2, L=128, seed=2025,
    )
    alg_cfg = AlgorithmConfig(outer_iters=15, inner_iters=50)

    structures = ["fully_connected","subarray"]
    out = {"structures": {}, "alg_cfg": vars(alg_cfg)}
    
    for struct in structures:
        cfg = copy.deepcopy(sys_cfg_base)
        cfg.rhs_structure = struct
        rng = np.random.default_rng(cfg.seed)
        scen = generate_scenario(cfg, rng)
        s1 = solve_SOOP1(scen, cfg, alg_cfg)
        s2 = solve_SOOP2(scen, cfg, alg_cfg)
        m = solve_MOOP(scen, cfg, alg_cfg,
                       soop1_result=s1, soop2_result=s2)
        out["structures"][struct] = {
            "SOOP1_history": s1["history"],
            "SOOP2_history": s2["history"],
            "MOOP_history": m["history"],
            "R_star": s1["history"]["sum_rate"][-1],
            "I_star": s2["history"]["sensing_mi"][-1],
        }
        print(f" {struct}: SOOP1 R={s1['history']['sum_rate'][-1]:.3f}, "
              f"SOOP2 I={s2['history']['sensing_mi'][-1]:.3f}, "
              f"MOOP (R,I)=({m['history']['sum_rate'][-1]:.3f},"
              f" {m['history']['sensing_mi'][-1]:.3f})")

    ts = time.strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"exp_convergence_{ts}.json"
    with open(path, "w") as f:
        json.dump(_to_jsonable(out), f, indent=2)
    print(f"[exp_convergence] saved -> {path}")


if __name__ == "__main__":
    main()
