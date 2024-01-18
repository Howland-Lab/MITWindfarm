from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mitwindfarm import Plotting
from mitwindfarm.Layout import Layout
from mitwindfarm.windfarm import Windfarm

FIGDIR = Path(__file__).parent.parent / "fig"
FIGDIR.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    windfarm = Windfarm()

    setpoints = [
        (2, 0),
        (2, np.deg2rad(30)),
        (2, 0),
    ]

    layout = Layout([0, 12, 24], [0, 0, 0])

    windfarm_sol = windfarm(layout, setpoints)
    windfarm_sol_rotated = windfarm(layout.rotate(5), setpoints)

    fig, axes = plt.subplots(2)
    Plotting.plot_windfarm(windfarm_sol, axes[0])
    Plotting.plot_windfarm(windfarm_sol_rotated, axes[1])

    plt.savefig(FIGDIR / "example_05_basic_windfarm.png", dpi=300, bbox_inches="tight")
