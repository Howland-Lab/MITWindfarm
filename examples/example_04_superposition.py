from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mitwindfarm import Superposition, RotorSolution, GaussianWakeModel, Uniform, Layout

superposition_methods = {
    "Linear": Superposition.Linear(),
    "Niayifar":Superposition.Niayifar(),
    "Quadratic": Superposition.Quadratic(),
    "Dominant": Superposition.Dominant(),
}

FIGDIR = Path(__file__).parent.parent / "fig"
FIGDIR.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    layout = Layout([0, 2, 4, 2, 0], [0, 0.5, 1, -2, 2])
    yaw, tilt, Cp, Ct, Ctprime, an, w4, REWS = 0, 0, 0, 0, 0, 0.3, 0.0, 1.0
    rotor_sols = [  # rotor solutions with varying u4 and v4
        RotorSolution(yaw, Cp, Ct, Ctprime, an, 0.5, 0.1, REWS, tilt = tilt, w4 = w4),
        RotorSolution(yaw, Cp, Ct, Ctprime, an, 0.5, -0.1, REWS, tilt = tilt, w4 = w4),
        RotorSolution(yaw, Cp, Ct, Ctprime, an, 0.5, 0.1, REWS, tilt = tilt, w4 = w4),
        RotorSolution(yaw, Cp, Ct, Ctprime, an, 0.5, 0.5, REWS, tilt = tilt, w4 = w4),
        RotorSolution(yaw, Cp, Ct, Ctprime, an, 0.8, -0.5, REWS, tilt = tilt, w4 = w4),
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
            yc, zc = wake.centerline(__x)
            ax.plot(__x, yc, ":r", lw=0.5)

        ax.set_title(label)
    fig.subplots_adjust(hspace=0.5)
        
    plt.savefig(FIGDIR / "example_04_superposition.png", dpi=300, bbox_inches="tight")
