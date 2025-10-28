from pathlib import Path

import matplotlib.pyplot as plt
from scipy.optimize import minimize

from mitwindfarm import Plotting
from mitwindfarm.windfarm import Windfarm, WindfarmSolution, Layout
from mitwindfarm.Rotor import BEM
from MITRotor.Momentum import UnifiedMomentum
from MITRotor.ReferenceTurbines import IEA15MW
# from rich import print
import numpy as np

FIGDIR = Path(__file__).parent.parent / "fig"
FIGDIR.mkdir(exist_ok=True, parents=True)

momentum_model= UnifiedMomentum(averaging = "rotor")
windfarm = Windfarm(
    rotor_model=BEM(IEA15MW(), momentum_model = momentum_model),
    TIamb=0.1,
)
layout = Layout([0, 12, 24], [0.0, 0.5, 1.0], [0.0, 0.0, 0.0])

def solve_for_setpoints(x) -> WindfarmSolution:
    setpoints = [(x[0], x[1], x[2], 0.0), (x[3], x[4], x[5], 0.0), (x[6], x[7], x[8], 0.0)]
    return windfarm(layout, setpoints)

def objective_func(x):
    windfarm_sol = solve_for_setpoints(x)
    return -windfarm_sol.Cp

if __name__ == "__main__":
    nturbs = 3
    angle_bounds = (-np.deg2rad(15), np.deg2rad(15))
    tsr_bounds = (3, 10)
    x0 = [0, 7, 0] * nturbs

    sol = minimize(
        objective_func,
        x0,
        bounds=[angle_bounds,tsr_bounds,angle_bounds] * nturbs,
    )
    print(sol)

    windfarm_sol_ref = solve_for_setpoints(x0)
    windfarm_sol_opt = solve_for_setpoints(sol.x)
    print(windfarm_sol_opt)

    fig, axes = plt.subplots(2, 1)
    Plotting.plot_windfarm(windfarm_sol_ref, axes[0])
    Plotting.plot_windfarm(windfarm_sol_opt, axes[1])
    fig.suptitle("Original and Optimized Wind Farm Layout for Yaw ($z = 0$)")
    axes[0].set_xlabel("$x/D$")
    axes[1].set_xlabel("$x/D$")
    axes[0].set_ylabel("$y/D$")
    axes[1].set_ylabel("$y/D$")

    axes[0].set_title(f"$C_P$: {windfarm_sol_ref.Cp:2.3f}")
    axes[1].set_title(
        f"$C_P$: {windfarm_sol_opt.Cp:2.3f} ({100*(windfarm_sol_opt.Cp/windfarm_sol_ref.Cp - 1):+2.1f}%)"
    )
    plt.tight_layout()
    plt.savefig(
        FIGDIR / "example_07_BEM_yaw_optimisation.png", dpi=300, bbox_inches="tight"
    )
