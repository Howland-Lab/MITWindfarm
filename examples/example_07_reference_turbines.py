from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mitwindfarm import Plotting
from mitwindfarm.Layout import Layout
from mitwindfarm.windfarm import RefWindfarm, RefTurbine


FIGDIR = Path(__file__).parent.parent / "fig"
FIGDIR.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    ws_lookup = np.linspace(0, 25,100)
    ctp_lookup = 2.0 * np.ones_like(ws_lookup)
    ctp_lookup[0:19] = 0
    turbine = RefTurbine(ws_lookup, ctp_lookup)
    windfarm = RefWindfarm(turbine, dim_freestream_vel=10)

    layout = Layout([0, 12, 24], [0, 0, 0])

    windfarm_sol = windfarm(layout)

    fig, axes = plt.subplots(2)
    Plotting.plot_windfarm(windfarm_sol, axes[0])
    plt.plot(np.arange(0,3) + 1, [rotor.REWS for rotor in windfarm_sol.rotors])

    plt.savefig(FIGDIR / "example_07_reference_turbines.png", dpi=300, bbox_inches="tight")