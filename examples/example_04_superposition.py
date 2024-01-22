from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mitwindfarm import Superposition
from mitwindfarm.Layout import Layout
from mitwindfarm.Rotor import RotorSolution
from mitwindfarm.Wake import GaussianWakeModel
from mitwindfarm.Windfield import Uniform

superposition_methods = {
    "Linear": Superposition.Linear(),
    "Quadratic": Superposition.Quadratic(),
    "Dominant": Superposition.Dominant(),
}

FIGDIR = Path(__file__).parent.parent / "fig"
FIGDIR.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    layout = Layout([0, 2, 4, 2, 0], [0, 0.5, 1, -2, 2])

    rotor_sols = [
        RotorSolution(0, 0, 0, 0, 0, 0.5, 0.1, 1.0),
        RotorSolution(0, 0, 0, 0, 0, 0.5, -0.1, 1.0),
        RotorSolution(0, 0, 0, 0, 0, 0.5, 0.1, 1.0),
        RotorSolution(0, 0, 0, 0, 0, 0.5, 0.5, 1.0),
        RotorSolution(0, 0, 0, 0, 0, 0.8, -0.5, 1.0),
    ]
    wake_model = GaussianWakeModel()
    wakes = [wake_model(x, y, z, sol) for (x, y, z), sol in zip(layout, rotor_sols)]

    base_windfield = Uniform()

    fig, axes = plt.subplots(len(superposition_methods), 1, sharex=True)

    _x, _y = np.linspace(-1, 12, 400), np.linspace(-3, 3, 401)
    xmesh, ymesh = np.meshgrid(_x, _y)

    for (label, method), ax in zip(superposition_methods.items(), axes):
        windfield = method(base_windfield, wakes)

        wsp = windfield.wsp(xmesh, ymesh, np.zeros_like(xmesh))

        ax.imshow(
            wsp,
            extent=[_x.min(), _x.max(), _y.min(), _y.max()],
            vmin=0,
            vmax=1,
            origin="lower",
        )
        for wake in wakes:
            __x = _x[_x > wake.x]
            ax.plot(__x, wake.centerline(__x), ":r", lw=0.5)

        ax.set_title(label)

    plt.savefig(FIGDIR / "example_04_superposition.png", dpi=300, bbox_inches="tight")
