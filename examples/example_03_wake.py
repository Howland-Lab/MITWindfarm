from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mitwindfarm import RotorSolution, GaussianWakeModel

FIGDIR = Path(__file__).parent.parent / "fig"
FIGDIR.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    yaw, tilt, Cp, Ct, Ctprime, an, u4, v4, w4, REWS = 0, 0, 0, 0, 0, 0.3, 0.5, 0.1, 0.0, 1.0
    rotor_sol = RotorSolution(yaw, tilt, Cp, Ct, Ctprime, an, u4, v4, w4, REWS)

    wake_model = GaussianWakeModel()
    wake = wake_model(1, 1, 0, rotor_sol, TIamb=0.1)

    _x, _y, _z = np.linspace(-1, 10, 400), np.linspace(-3, 3, 400), np.linspace(-3, 3, 400)
    xmesh, ymesh, zmesh = np.meshgrid(_x, _y, _z)

    deficit = wake.deficit(xmesh, ymesh, zmesh)
    y_centerline, z_centerline = wake.centerline(_x)

    fig, (ax0, ax1) = plt.subplots(1, 2)
    # plot x-y plane
    ax0.imshow(deficit[:, :, 200], extent=[_x.min(), _x.max(), _y.min(), _y.max()], origin="lower")
    ax0.plot(_x, y_centerline, "--r")
    # plot x-z plane
    ax1.imshow(deficit[:, 200, :], extent=[_x.min(), _x.max(), _z.min(), _z.max()], origin="lower")
    ax1.plot(_x, z_centerline, "--r")

    plt.savefig(FIGDIR / "example_03_wake.png", dpi=300, bbox_inches="tight")
