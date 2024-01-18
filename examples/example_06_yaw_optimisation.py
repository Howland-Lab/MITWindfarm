from pathlib import Path

import matplotlib.pyplot as plt
from scipy.optimize import minimize

from mitwindfarm import Plotting
from mitwindfarm.Layout import Layout
from mitwindfarm.windfarm import Windfarm, WindfarmSolution

FIGDIR = Path(__file__).parent.parent / "fig"
FIGDIR.mkdir(exist_ok=True, parents=True)

windfarm = Windfarm()
layout = Layout([0, 12, 24], [0.0, 0.5, 1.0])


def solve_for_setpoints(x) -> WindfarmSolution:
    setpoints = [(2, x[0]), (2, x[1]), (2, x[2])]
    return windfarm(layout, setpoints)


def objective_func(x):
    windfarm_sol = solve_for_setpoints(x)
    return -windfarm_sol.Cp()


if __name__ == "__main__":
    x0 = [0, 0, 0]
    sol = minimize(objective_func, x0)

    windfarm_sol_ref = solve_for_setpoints(x0)
    windfarm_sol_opt = solve_for_setpoints(sol.x)

    fig, axes = plt.subplots(2, 1)
    Plotting.plot_windfarm(windfarm_sol_ref, axes[0])
    Plotting.plot_windfarm(windfarm_sol_opt, axes[1])

    axes[0].set_title(f"$C_P$: {windfarm_sol_ref.Cp():2.3f}")
    axes[1].set_title(
        f"$C_P$: {windfarm_sol_opt.Cp():2.3f} ({100*(windfarm_sol_opt.Cp()/windfarm_sol_ref.Cp() - 1):+2.1f}%)"
    )

    plt.savefig(
        FIGDIR / "example_06_yaw_optimisation.png", dpi=300, bbox_inches="tight"
    )
