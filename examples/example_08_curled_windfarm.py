from pathlib import Path

import matplotlib.pyplot as plt
from mitwindfarm import Uniform, Layout, PowerLaw
from mitwindfarm.Plotting import plot_windfarm
from mitwindfarm.windfarm import Windfarm, CurledWindfarm
from mitwindfarm.Rotor import UnifiedAD_TI
import numpy as np
import time

FIGDIR = Path(__file__).parent.parent / "fig"
FIGDIR.mkdir(exist_ok=True, parents=True)


def plot_example(powerlaw=False):
    if powerlaw: 
        zhub = 1  # hub height in diameters
        base_windfield = PowerLaw(Uref=1, zref=zhub, exp=0.11, TIamb=0.05)
    else:  # plot uniform inflow
        zhub = 0
        base_windfield = Uniform(TIamb=0.05)  # 5% ambient TI

    wf_curled = CurledWindfarm(
        rotor_model=UnifiedAD_TI(),
        base_windfield=base_windfield,
        solver_kwargs=dict(
            dy=1 / 10,
            dz=1 / 10,
            integrator="scipy_rk23",  # see mitwindfarm.utils.integrate
            k_model="k-l",  # alternatives: "const", "2021"
            verbose=False,
        ),
    )
    wf_gauss = Windfarm(TIamb=0.05)  # 5% ambient TI, default model is Gaussian wake
    layout = Layout([0, 5, 10], [0, 0.4, 0.8], [zhub, zhub, zhub])  # non-dim by diameter D
    setpoints = [  # for UnifiedAD_TI() rotor, set points are (Ctprime, yaw, tilt) tuple pairs
        (2, np.radians(30), np.radians(0)),
        (2, np.radians(15), np.radians(0)),
        (2, 0, 0),
    ]  #  Example setpoints for three turbines

    # compute windfarm solutions (Cp)
    wf_solutions = []
    for name, _wf in zip(["Curl", "Gauss"], [wf_curled, wf_gauss]):
        time_st = time.time()
        sol = _wf(layout, setpoints)
        print(f"Windfarm solution for {name} in {time.time() - time_st:.2f} seconds")
        wf_solutions.append((name, sol))

    # plot the comparison: wind speed (streamwise slice) and power
    fig1, axarr1 = plt.subplots(
        figsize=(4 * len(wf_solutions), 6),
        nrows=3,
        ncols=len(wf_solutions),
        sharex=True,
        sharey="row",
        height_ratios=(1, 1, 2),
    )
    for axs, (name, sol) in zip(axarr1.T, wf_solutions):
        plot_windfarm(sol, ax=axs[0], z = zhub, pad=2, axis=True)
        y_plot_val = 0
        plot_windfarm(sol, ax=axs[1], y = y_plot_val, pad=2, axis=True)
        axs[0].set_xlabel("$x/D$")
        axs[1].set_xlabel("$x/D$")
        axs[0].set_title(name + f" at z = {zhub}")
        axs[1].set_title(name + f" at y = {y_plot_val}")
        # plot power per turbine
        axs[2].bar(layout.x, [r.Cp for r in sol.rotors], width=2)
        if np.all(axs == (axarr1.T)[0]):  # only label the first columns
            axs[0].set_ylabel("$y/D$")
            axs[1].set_ylabel("$z/D$")
            axs[2].set_ylabel("$C_P$")
        axs[2].set_xticks(layout.x)
        axs[2].set_xticklabels(np.arange(len(layout.x)) + 1)
        axs[2].set_xlabel("Turbine row")

    plt.savefig(FIGDIR / f"{Path(__file__).stem}1.png", bbox_inches="tight")

    fig2, axarr2 = plt.subplots(
        figsize=(4 * len(wf_solutions), 4),
        nrows=1,
        ncols=len(wf_solutions),
        sharex=True,
        sharey="row",
    )
    for axs, (name, sol) in zip(axarr2.T, wf_solutions):
        # at first turbine's hub
        x_plot_val = 4
        plot_windfarm(sol, ax=axs, x = x_plot_val, pad=2, axis=True)
        axs.set_title(name + f" at x = {x_plot_val}")
        axs.set_xlabel("$y/D$")
        axs.set_ylabel("$z/D$")

    
    plt.savefig(FIGDIR / f"{Path(__file__).stem}2.png", bbox_inches="tight")

if __name__ == "__main__":
    plot_example(powerlaw=False)
    plt.close()
