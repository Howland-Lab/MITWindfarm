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

    wf = CurledWindfarm(
        rotor_model=UnifiedAD_TI(),
        base_windfield=base_windfield,
        solver_kwargs=dict(
            dy=1 / 10,
            dz=1 / 10,
            integrator="scipy_rk23",  # see
            k_model="k-l",  # alternatives: "const", "2021"
            verbose=False,
        ),
    )
    wf_gauss = Windfarm(TIamb=0.05)
    layout = Layout([0, 5, 10], [0, 0.4, 0.8], [zhub, zhub, zhub])
    setpoints = [
        (2, np.radians(30)),
        (2, np.radians(15)),
        (2, 0),
    ]  #  Example setpoints for two turbines

    wf_solutions = []
    for name, _wf in zip(["Curl", "Gauss"], [wf, wf_gauss]):
        time_st = time.time()
        sol = _wf(layout, setpoints)
        print(f"Windfarm solution for {name} in {time.time() - time_st:.2f} seconds")
        wf_solutions.append((name, sol))

    fig, axarr = plt.subplots(
        figsize=(4 * len(wf_solutions), 4),
        nrows=2,
        ncols=len(wf_solutions),
        sharex=True,
        sharey="row", 
        height_ratios=(1, 2),
    )
    for axs, (name, sol) in zip(axarr.T, wf_solutions):
        plot_windfarm(sol, ax=axs[0], pad=2, axis=True)
        axs[0].set_xlabel("$x/D$")
        axs[0].set_title(name)
        # plot power per turbine
        axs[1].bar(layout.x, [r.Cp for r in sol.rotors], width=2)
        if np.all(axs == (axarr.T)[0]):  # only label the first columns
            axs[0].set_ylabel("$y/D$")
            axs[1].set_ylabel("$C_P$")
        axs[1].set_xticks(layout.x)
        axs[1].set_xticklabels(np.arange(len(layout.x)) + 1)
        axs[1].set_xlabel("Turbine row")

    plt.savefig(FIGDIR / f"{Path(__file__).stem}.png", bbox_inches="tight")


if __name__ == "__main__":
    plot_example(powerlaw=True)
    plt.close()
