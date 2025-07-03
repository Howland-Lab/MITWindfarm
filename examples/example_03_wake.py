from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mitwindfarm import RotorSolution, GaussianWakeModel

FIGDIR = Path(__file__).parent.parent / "fig"
FIGDIR.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    # create a rotor solution
    yaw, tilt, Cp, Ct, Ctprime, an, u4, v4, w4, REWS = 0, 0, 0, 0, 0, 0.3, 0.5, 0.1, 0.1, 1.0
    rotor_sol = RotorSolution(yaw, tilt, Cp, Ct, Ctprime, an, u4, v4, w4, REWS)
    # create a wake model
    wake_model = GaussianWakeModel()
    xturb, yturb, zturb = 1, 0, 0
    wake = wake_model(xturb, yturb, zturb, rotor_sol, TIamb=0.1)
    # define grid / turbine index
    npoints = 50
    _x, _y, _z = np.linspace(-1, 10, npoints), np.linspace(-3, 3, npoints), np.linspace(-3, 3, npoints)
    xturb_idx = np.abs(_x - xturb).argmin()
    yturb_idx = np.abs(_y - yturb).argmin()
    zturb_idx = np.abs(_z - zturb).argmin()
    # fine centerline
    y_centerline, z_centerline = wake.centerline(_x)
    # define meshgrids along planes and calculate the deficits
    x_mesh, y_mesh = np.meshgrid(_x, _y)
    xy_deficit = wake.deficit(x_mesh, y_mesh, zturb) # x-y plane 

    x_mesh, z_mesh = np.meshgrid(_x, _z)
    xz_deficit = wake.deficit(x_mesh, yturb, z_mesh) # x-z plane 

    y_mesh, z_mesh = np.meshgrid(_y, _z)
    yz_deficit = wake.deficit(xturb, y_mesh, z_mesh) # y-z plane 

    # plot the deficits and wake center lines
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    vmin = min(np.min(xy_deficit), np.min(xz_deficit), np.min(yz_deficit))
    vmax = max(np.max(xy_deficit), np.max(xz_deficit), np.max(yz_deficit))
    # plot x-y plane
    ax0.imshow(xy_deficit, extent=[_x.min(), _x.max(), _y.min(), _y.max()], origin="lower", vmin = vmin, vmax = vmax)
    ax0.plot(_x, y_centerline, "--r")
    # plot x-z plane
    ax1.imshow(xz_deficit, extent=[_x.min(), _x.max(), _z.min(), _z.max()], origin="lower", vmin = vmin, vmax = vmax)
    ax1.plot(_x, z_centerline, "--r")
    # # plot y-z plane
    im2 = ax2.imshow(yz_deficit, extent=[_y.min(), _y.max(), _z.min(), _z.max()], origin="lower", vmin = vmin, vmax = vmax)
    # add colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im2, cax=cbar_ax)
    # save figure
    plt.savefig(FIGDIR / "example_03_wake.png", dpi=300, bbox_inches="tight")
