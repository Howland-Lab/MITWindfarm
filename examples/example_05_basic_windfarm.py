from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
from MITRotor import IEA15MW

from mitwindfarm import BEM, Plotting, Windfarm, Layout

FIGDIR = Path(__file__).parent.parent / "fig"
FIGDIR.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    windfarm = Windfarm(rotor_model=BEM(IEA15MW()))

    setpoints = [ # pitch, tsr, yaw, tilt
        (0, 7, 0, np.deg2rad(20)),
        (0, 7, np.deg2rad(10), 0),
        (0, 7, 0, 0),
    ]

    layout = Layout([0, 12 / 2, 24 / 2], [0, 0, 0])

    windfarm_sol = windfarm(layout, setpoints)
    windfarm_sol_rotated = windfarm(layout.rotate(5), setpoints)

    fig, axes = plt.subplots(2)
    Plotting.plot_windfarm(windfarm_sol, axes[0])
    Plotting.plot_windfarm(windfarm_sol_rotated, axes[1])

    plt.savefig(FIGDIR / "example_05_basic_windfarm.png", dpi=300, bbox_inches="tight")
