import matplotlib.pyplot as plt
import numpy as np

from .windfarm import WindfarmSolution


def plot_windfarm(sol: WindfarmSolution, ax=None, pad=1, z=None, frame=True, axis=False, res: int=400):
    if ax is None:
        _, ax = plt.subplots()

    xlim = (np.min(sol.layout.x) - pad, np.max(sol.layout.x) + 10)
    ylim = (np.min(sol.layout.y) - pad, np.max(sol.layout.y) + pad)
    z = np.mean(sol.layout.z) if z is None else z

    # plot windfield and turbine stats
    _x, _y = np.linspace(*xlim, res), np.linspace(*ylim, res)
    xmesh, ymesh = np.meshgrid(_x, _y)

    wsp = sol.windfield.wsp(xmesh, ymesh, np.full_like(xmesh, z))
    ax.imshow(
        wsp, extent=[*xlim, *ylim], vmin=0, vmax=2, origin="lower", cmap="RdYlBu_r"
    )

    for (turb_x, turb_y, _), rotor in zip(sol.layout, sol.rotors):
        # Draw turbine
        R = 0.5
        p = np.array([[0, 0], [+R, -R]])
        rotmat = np.array(
            [
                [np.cos(rotor.yaw), -np.sin(rotor.yaw)],
                [np.sin(rotor.yaw), np.cos(rotor.yaw)],
            ]
        )

        p = rotmat @ p + np.array([[turb_x], [turb_y]])

        ax.plot(p[0, :], p[1, :], "k", lw=2)
    ax.set(frame_on=frame)
    if axis is False:
        ax.set(xticks=[], yticks=[])
