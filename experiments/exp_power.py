"""
Experiment: sum-rate / sensing-MI versus transmit power Pt (dBm).
"""
from config import SystemConfig, AlgorithmConfig
from experiments.runner import run_sweep


def main():
    sys_cfg = SystemConfig(
        Mt_h=16, Mt_v=16, Mr_h=16, Mr_v=16, N=8,
        Kc=3, Ks=2, Ke=2, L=128, seed=2025,
        rhs_structure="subarray",
    )
    alg_cfg = AlgorithmConfig(outer_iters=8, inner_iters=15)

    pt_list = [10, 15, 20, 25, 30, 35]   # dBm
    omega_list = [(0.2, 0.8), (0.5, 0.5), (0.8, 0.2)]

    run_sweep("exp_power", sys_cfg, alg_cfg,
              param_name="Pt_dBm", param_values=pt_list,
              n_trials=3,
              methods=("SOOP1", "SOOP2", "MOOP"),
              omega_list=omega_list)


if __name__ == "__main__":
    main()
