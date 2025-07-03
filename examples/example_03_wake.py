from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mitwindfarm import RotorSolution, GaussianWakeModel

FIGDIR = Path(__file__).parent.parent / "fig"
FIGDIR.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    yaw, tilt, Cp, Ct, Ctprime, an, u4, v4, w4, REWS = 0, 0, 0, 0, 0, 0.3, 0.5, 0.0, 0.1, 1.0
    rotor_sol = RotorSolution(yaw, tilt, Cp, Ct, Ctprime, an, u4, v4, w4, REWS)

    wake_model = GaussianWakeModel()
    wake = wake_model(1, 1, 0, rotor_sol, TIamb=0.1)

    npoints = 400
    _x, _y, _z = np.linspace(-1, 10, npoints), np.linspace(-3, 3, npoints), np.linspace(-3, 3, npoints)
    # xmesh, ymesh, zmesh = np.meshgrid(_x, _y, _z)


    deficit = wake.deficit(_x, _y, _z)
    ds = deficit.shape

    y_centerline, z_centerline = wake.centerline(_x)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    # plot x-y plane
    ax0.imshow(deficit[:, :, round(npoints / 2)], extent=[_x.min(), _x.max(), _y.min(), _y.max()], origin="lower")
    ax0.plot(_x, y_centerline, "--r")
    # plot x-z plane
    ax1.imshow(deficit[:, round(npoints / 2), :], extent=[_x.min(), _x.max(), _z.min(), _z.max()], origin="lower")
    ax1.plot(_x, z_centerline, "--r")
    # plot y-z plane
    ax2.imshow(deficit[round(npoints / 2), :, :], extent=[_x.min(), _x.max(), _z.min(), _z.max()], origin="lower")

    plt.savefig(FIGDIR / "example_03_wake.png", dpi=300, bbox_inches="tight")
