from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mitwindfarm import RotorSolution, GaussianWakeModel

FIGDIR = Path(__file__).parent.parent / "fig"
FIGDIR.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    yaw, tilt, Cp, Ct, Ctprime, an, u4, v4, w4, REWS = 0, 0, 0, 0, 0, 0.3, 0.5, 0.2, 0.0, 1.0
    rotor_sol = RotorSolution(yaw, tilt, Cp, Ct, Ctprime, an, u4, v4, w4, REWS)

    wake_model = GaussianWakeModel()
    xturb, yturb, zturb = 1, 0, 0
    wake = wake_model(xturb, yturb, zturb, rotor_sol, TIamb=0.1)

    npoints = 50
    _x, _y, _z = np.linspace(-1, 10, npoints), np.linspace(-3, 3, npoints), np.linspace(-3, 3, npoints)
    xturb_idx = np.abs(_x - xturb).argmin()
    yturb_idx = np.abs(_y - yturb).argmin()
    zturb_idx = np.abs(_z - zturb).argmin()

    deficit = wake.deficit(_x, _y, _z)
    ds = deficit.shape

    y_centerline, z_centerline = wake.centerline(_x)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    vmin = np.min(deficit)
    vmax = np.max(deficit)
    # plot x-y plane
    ax0.imshow(deficit[:, :, zturb_idx], extent=[_x.min(), _x.max(), _y.min(), _y.max()], origin="lower", vmin = vmin, vmax = vmax)
    ax0.plot(_x, y_centerline, "--r")
    # plot x-z plane
    ax1.imshow(np.transpose(deficit[yturb_idx, :, :]), extent=[_x.min(), _x.max(), _z.min(), _z.max()], origin="lower")
    ax1.plot(_x, z_centerline, "--r")
    # # plot y-z plane
    im2 = ax2.imshow(deficit[:, xturb_idx, :], extent=[_y.min(), _y.max(), _z.min(), _z.max()], origin="lower")
    # add colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im2, cax=cbar_ax)
    # save figure
    plt.savefig(FIGDIR / "example_03_wake.png", dpi=300, bbox_inches="tight")
