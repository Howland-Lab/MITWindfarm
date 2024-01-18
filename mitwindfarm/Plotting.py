import matplotlib.pyplot as plt
import numpy as np

from .windfarm import WindfarmSolution


def plot_windfarm(sol: WindfarmSolution, ax=None, pad=1, frame=True, axis=False):
    if ax is None:
        _, ax = plt.subplots()

    xlim = (sol.layout.x.min() - pad, sol.layout.x.max() + 10)
    ylim = (sol.layout.y.min() - pad, sol.layout.y.max() + pad)

    # plot windfield and turbine stats
    _x, _y = np.linspace(*xlim, 400), np.linspace(*ylim, 401)
    xmesh, ymesh = np.meshgrid(_x, _y)

    wsp = sol.windfield.wsp(xmesh, ymesh, np.zeros_like(xmesh))
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
