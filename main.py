"""
Top-level entry point. Run individual experiments via:

    python main.py smoke          # quick sanity check (small problem)
    python main.py power          # exp_power.py
    python main.py arraysize      # exp_arraysize.py
    python main.py userscale      # exp_userscale.py (Kc & Ks)
    python main.py pareto         # exp_pareto.py
    python main.py convergence    # exp_convergence.py
    python main.py all            # everything except smoke
"""
import sys
import numpy as np

from config import SystemConfig, AlgorithmConfig
from system_model import generate_scenario
from algorithms import solve_SOOP1, solve_SOOP2, solve_MOOP


def smoke_test():
    """A quick small-scale run that exercises every module."""
    print("=== SMOKE TEST ===")
    sys_cfg = SystemConfig(
        Mt_h=8, Mt_v=8,         # 64 elements only
        Mr_h=8, Mr_v=8,
        N=4, Kc=2, Ks=2, Ke=1,
        L=64, seed=42,
        rhs_structure="subarray",
    )
    alg_cfg = AlgorithmConfig(outer_iters=4, inner_iters=8)

    rng = np.random.default_rng(sys_cfg.seed)
    scen = generate_scenario(sys_cfg, rng=rng)
    print(f"scenario built: Mt={scen.Mt}, Mr={scen.Mr}, "
          f"Kc={scen.Kc}, Ks={scen.Ks}, Ke={scen.Ke}")
    print(f"Phi shape: {scen.Phi.shape},  H shape: {scen.H.shape}")

    s1 = solve_SOOP1(scen, sys_cfg, alg_cfg)
    print(f"SOOP1 done. final R={s1['history']['sum_rate'][-1]:.3f} bps/Hz, "
          f"I={s1['history']['sensing_mi'][-1]:.3f}")

    s2 = solve_SOOP2(scen, sys_cfg, alg_cfg)
    print(f"SOOP2 done. final R={s2['history']['sum_rate'][-1]:.3f}, "
          f"I={s2['history']['sensing_mi'][-1]:.3f}")

    m = solve_MOOP(scen, sys_cfg, alg_cfg,
                   soop1_result=s1, soop2_result=s2,
                   omega1=0.5, omega2=0.5)
    print(f"MOOP done.  final R={m['history']['sum_rate'][-1]:.3f}, "
          f"I={m['history']['sensing_mi'][-1]:.3f}, "
          f"R*={m['R_star']:.3f}, I*={m['I_star']:.3f}")
    print("=== smoke test passed ===")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    target = sys.argv[1]

    if target == "smoke":
        smoke_test()
    elif target == "power":
        from experiments.exp_power import main as run; run()
    elif target == "arraysize":
        from experiments.exp_arraysize import main as run; run()
    elif target == "userscale":
        from experiments.exp_userscale import sweep_Kc, sweep_Ks
        sweep_Kc(); sweep_Ks()
    elif target == "pareto":
        from experiments.exp_pareto import main as run; run()
    elif target == "convergence":
        from experiments.exp_convergence import main as run; run()
    elif target == "all":
        from experiments.exp_power import main as run1
        from experiments.exp_arraysize import main as run2
        from experiments.exp_userscale import sweep_Kc, sweep_Ks
        from experiments.exp_pareto import main as run4
        from experiments.exp_convergence import main as run5
        run1(); run2(); sweep_Kc(); sweep_Ks(); run4(); run5()
    else:
        print(f"Unknown target: {target}")
        print(__doc__)


if __name__ == "__main__":
    main()
