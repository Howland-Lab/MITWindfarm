from pathlib import Path

import matplotlib.pyplot as plt
from scipy.optimize import minimize

from mitwindfarm import Plotting
from mitwindfarm.windfarm import Windfarm, WindfarmSolution, Layout
from mitwindfarm.Rotor import BEM
from MITRotor.Momentum import UnifiedMomentum
from MITRotor.ReferenceTurbines import IEA15MW
from UnifiedMomentumModel.Utilities.Geometry import calc_eff_yaw
# from rich import print
import numpy as np

FIGDIR = Path(__file__).parent.parent / "fig"
FIGDIR.mkdir(exist_ok=True, parents=True)

momentum_model= UnifiedMomentum(averaging = "rotor")
windfarm = Windfarm(
    rotor_model=BEM(IEA15MW(), momentum_model = momentum_model),
    TIamb=0.1,
)
layout = Layout([0, 12], [0.0, 0.0], [0.0, 0.0])

def solve_for_setpoints(x) -> WindfarmSolution:
    setpoints = [(x[0], x[1], x[2], 0.0), (x[3], x[4], x[5], 0.0), (x[6], x[7], x[8], 0.0)]
    return windfarm(layout, setpoints)

def objective_func(x):
    windfarm_sol = solve_for_setpoints(x)
    return -windfarm_sol.Cp

if __name__ == "__main__":
    pitch_val = 0
    tsr_val = 7

    yaw_tilt_val = np.deg2rad(15)
    eff_yaw = calc_eff_yaw(yaw_tilt_val, yaw_tilt_val)
    yaw_val = eff_yaw
    tilt_val = eff_yaw

    aligned_setpoints = (pitch_val, tsr_val, 0, 0)
    setpoints_yaw = [(pitch_val, tsr_val, eff_yaw, 0), aligned_setpoints]
    setpoints_tilt = [(pitch_val, tsr_val, 0, eff_yaw), aligned_setpoints]
    setpoints_yaw_and_tilt = [(pitch_val, tsr_val, yaw_tilt_val, yaw_tilt_val), aligned_setpoints]   

    yaw_sol = windfarm(layout, setpoints_yaw)
    tilt_sol = windfarm(layout, setpoints_tilt) 
    yaw_and_tilt_sol = windfarm(layout, setpoints_yaw_and_tilt) 

    fig, axes = plt.subplots(2, 3)
    Plotting.plot_windfarm(yaw_sol, axes[0, 0], z = 0)
    Plotting.plot_windfarm(tilt_sol, axes[0, 1], z = 0)
    Plotting.plot_windfarm(yaw_and_tilt_sol, axes[0, 2], z = 0)
    axes[0, 0].set_ylabel("$z/D$")

    Plotting.plot_windfarm(yaw_sol, axes[1, 0], y = 0)
    Plotting.plot_windfarm(tilt_sol, axes[1, 1], y = 0)
    Plotting.plot_windfarm(yaw_and_tilt_sol, axes[1, 2], y = 0)
    axes[1, 0].set_ylabel("$y/D$")
    axes[1, 0].set_xlabel("$x/D$")
    axes[1, 1].set_xlabel("$x/D$")
    axes[1, 2].set_xlabel("$x/D$")

    fig.suptitle("$C_P$ for Equivalent Turbine Setups", size = 16)
    deg_yaw_tilt, deg_eff_yaw = np.rad2deg(yaw_tilt_val), np.rad2deg(eff_yaw)
    axes[0, 0].set_title(f"Yaw {np.round(deg_eff_yaw, decimals=1)}$^\circ$\n$C_P$: {yaw_sol.Cp:2.3f}")
    axes[0, 1].set_title(f"Tilt {np.round(deg_eff_yaw, decimals=1)}$^\circ$\n$C_P$: {tilt_sol.Cp:2.3f}")
    axes[0, 2].set_title(f"Yaw {np.round(deg_yaw_tilt, decimals=1)}$^\circ$ & Tilt {np.round(deg_yaw_tilt, decimals=1)}$^\circ$\n$C_P$: {yaw_and_tilt_sol.Cp:2.3f}")
    plt.tight_layout()
    plt.savefig(
        FIGDIR / "example_09_yaw_tilt_comparison.png", dpi=300, bbox_inches="tight"
    )
