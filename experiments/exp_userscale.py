"""
Experiment: performance vs number of users / targets.
"""
from config import SystemConfig, AlgorithmConfig
from experiments.runner import run_sweep


def sweep_Kc():
    sys_cfg = SystemConfig(
        Mt_h=16, Mt_v=16, Mr_h=16, Mr_v=16, N=8,
        Ks=2, Ke=2, L=128, seed=2025,
        rhs_structure="subarray",
    )
    alg_cfg = AlgorithmConfig(outer_iters=8, inner_iters=15)
    Kc_list = [1, 2, 3, 4, 5]
    run_sweep("exp_userscale_Kc", sys_cfg, alg_cfg,
              param_name="Kc", param_values=Kc_list,
              n_trials=3,
              methods=("SOOP1", "SOOP2", "MOOP"),
              omega_list=[(0.5, 0.5)])


def sweep_Ks():
    sys_cfg = SystemConfig(
        Mt_h=16, Mt_v=16, Mr_h=16, Mr_v=16, N=8,
        Kc=3, Ke=2, L=128, seed=2025,
        rhs_structure="subarray",
    )
    alg_cfg = AlgorithmConfig(outer_iters=8, inner_iters=15)
    Ks_list = [1, 2, 3, 4]
    run_sweep("exp_targetscale_Ks", sys_cfg, alg_cfg,
              param_name="Ks", param_values=Ks_list,
              n_trials=3,
              methods=("SOOP1", "SOOP2", "MOOP"),
              omega_list=[(0.5, 0.5)])


if __name__ == "__main__":
    sweep_Kc()
    sweep_Ks()
