from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mitwindfarm.Rotor import RotorSolution
from mitwindfarm.Wake import GaussianWakeModel

FIGDIR = Path(__file__).parent.parent / "fig"
FIGDIR.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    rotor_sol = RotorSolution(0, 0, 0, 0, 0, 0.5, 0.1)


    wake_model = GaussianWakeModel()
    wake = wake_model(1, 1, 0, rotor_sol)

    _x, _y = np.linspace(-1, 10, 400), np.linspace(-3, 3, 400)
    xmesh, ymesh = np.meshgrid(_x, _y)

    deficit = wake.deficit(xmesh, ymesh)
    centerline = wake.centerline(_x)

    plt.imshow(deficit, extent=[_x.min(), _x.max(), _y.min(), _y.max()], origin="lower")

    plt.plot(_x, centerline, "--r")

    plt.savefig(FIGDIR / "example_03_wake.png", dpi=300, bbox_inches="tight")
