"""
In this example, we show an example for the TANDEM (Turbulence ANd DEficit Momentum)
wake model, which solves a coupled system of equations for the wake deficit and
wake-added TKE. The TANDEM model is implemented in the CurledWindfarm class, and is
included as a turbulence closure (`k_model`).

For more on the TANDEM model, see:
Heck and Howland, "The TANDEM wakemodel:coupled turbulence and deficit
momentum modeling in stratified atmospheric boundary layers" in Wind Energy
Science (preprint), https://wes.copernicus.org/preprints/wes-2026-149/
"""

from pathlib import Path

import matplotlib.pyplot as plt
from mitwindfarm import Uniform, GridLayout, PowerLaw
from mitwindfarm.Plotting import plot_windfarm
from mitwindfarm.windfarm import Windfarm, CurledWindfarm
from mitwindfarm.Rotor import UnifiedAD_TI
import numpy as np
import time

FIGDIR = Path(__file__).parent.parent / "fig"
FIGDIR.mkdir(exist_ok=True, parents=True)


def plot_example():
    base_windfield = Uniform(TIamb=0.05)  # 5% ambient TI

    wf = CurledWindfarm(
        rotor_model=UnifiedAD_TI(),
        base_windfield=base_windfield,
        solver_kwargs=dict(
            dy=0.05,
            dz=0.05,
            # u_model="upwind",  # "upwind" is recommended in yaw misalignment
            integrator="scipy_rk45",  # options: ef, scipy_rk23, scipy_rk45
            k_model="tandem",
            auto_expand=False,  # don't expand the grid automatically
        ),
    )
    wf_gauss = Windfarm(TIamb=0.05)  # 5% ambient TI, default model is Gaussian wake
    layout = GridLayout(6.0, 6.0, 4, 1)  # non-dim by diameter D
    # for UnifiedAD_TI() rotor, set points are (Ctprime, yaw, tilt [optional]) tuple pairs
    setpoints = [(4 / 3, np.radians(0))] + [
        (4 / 3, 0),
    ] * (len(layout) - 1)

    # compute windfarm solutions (Cp)
    wf_solutions = []
    time_st = time.time()
    sol = wf(layout, setpoints)
    print(f"Windfarm solved in {time.time() - time_st:.2f} seconds")

    # can manually extend integration to solve for more of the wake
    sol.windfield.march_to(25, 0, 0)  # march to x/D=25, y=0, z=0

    # extract fields to plot
    grid = sol.windfield.grid
    zid_hub = np.argmin(np.abs(grid[2]))  # hub height in diameters
    du = sol.windfield.du[..., zid_hub]
    dk = sol.windfield.dk[..., zid_hub]
    # eddy viscosity can be extracted in post as well:
    nu_T = sol.windfield.modules["dk"].postprocess_nu_T()[..., zid_hub]

    fig, axarr = plt.subplots(
        figsize=(4, 5), nrows=4, layout="constrained", height_ratios=[2, 1, 1, 1]
    )
    _r = np.arange(len(sol.rotors)) + 1
    axarr[0].plot(_r, [r.Cp for r in sol.rotors], marker="o", color="k")
    axarr[0].set_xticks(_r)
    axarr[0].set_xlabel("Row")
    axarr[0].set_ylabel("$C_P$")
    axarr[0].set_ylim([0, 16 / 27])

    # plot fields
    im = axarr[1].pcolormesh(grid[0], grid[1], du.T, cmap="viridis")
    plt.colorbar(im, ax=axarr[1], label="$\\overline{\\Delta u}/u_\\infty$")
    im = axarr[2].pcolormesh(grid[0], grid[1], dk.T, cmap="inferno")
    plt.colorbar(im, ax=axarr[2], label="$\\overline{\\Delta k}/u_\\infty^2$")
    im = axarr[3].pcolormesh(grid[0], grid[1], nu_T.T, cmap="cividis")
    plt.colorbar(im, ax=axarr[3], label="$\\nu_T/(D u_\\infty)$")
    for ax in axarr[1:]:
        ax.set_ylabel("$y/D$")
        ax.set_xticks([]) if ax != axarr[-1] else None
        ax.set_aspect(1.0)
    axarr[-1].set_xlabel("$x/D$")

    plt.savefig(FIGDIR / f"{Path(__file__).stem}.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    plot_example()
