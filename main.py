"""
Top-level entry point. Run individual experiments via:

    python main.py smoke               # quick sanity check (small problem)
    python main.py power               # exp_power.py
    python main.py arraysize           # exp_arraysize.py
    python main.py userscale           # exp_userscale.py (Kc & Ks)
    python main.py pareto              # exp_pareto.py
    python main.py convergence         # exp_convergence.py
    python main.py montecarlo          # exp_montecarlo.py (default 50 trials)
    python main.py montecarlo 100      # exp_montecarlo.py with 100 trials
    python main.py power_sensing       # exp_power_sensing.py (MI vs Pt)
    python main.py ks_sensing          # exp_ks_sensing.py (MI vs Ks)
    python main.py all                 # everything except smoke
"""
import sys
import numpy as np

from config import SystemConfig, AlgorithmConfig
from system_model import generate_scenario
from algorithms import solve_SOOP1, solve_SOOP2, solve_MOOP




def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    target = sys.argv[1]

    if target == "power_sensing":
        from experiments.exp_power_sensing import main as run; run()
    elif target == "ks_sensing":
        from experiments.exp_ks_sensing import main as run; run()
    elif target == "power":
        from experiments.exp_power import main as run; run()
    elif target == "arraysize":
        from experiments.exp_arraysize import main as run; run()
    elif target == "userscale":
        from experiments.exp_userscale import sweep_Kc, sweep_Ks
        sweep_Kc(); sweep_Ks()
    elif target == "pareto":
        from experiments.exp_pareto import main as run; run()
    elif target == "SOOP1_testing":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        from experiments.exp_SOOP1_testing import main as run; run(n)
    elif target == "SOOP2_testing":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        from experiments.exp_SOOP2_testing import main as run; run(n)
    elif target == "convergence":
        from experiments.exp_convergence import main as run; run()
    elif target == "montecarlo":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        from experiments.exp_montecarlo import main as run; run(n)
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
