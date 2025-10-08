import matplotlib.pyplot as plt
import numpy as np

from .windfarm import WindfarmSolution


def plot_windfarm(sol: WindfarmSolution, ax=None, pad=1, y=None, z=None, frame=True, axis=False, res: int=400):
    if ax is None:
        _, ax = plt.subplots()

    xlim = (np.min(sol.layout.x) - pad, np.max(sol.layout.x) + 10)
    ylim = (np.min(sol.layout.y) - pad, np.max(sol.layout.y) + pad)
    zlim = (np.min(sol.layout.z) - pad, np.max(sol.layout.z) + pad)
    xz_frame = False
    if y is not None:
        xz_frame = True
        _x, _z = np.linspace(*xlim, res), np.linspace(*zlim, res)
        xmesh, zmesh = np.meshgrid(_x, _z)
        wsp = sol.windfield.wsp(xmesh, np.full_like(xmesh, y), zmesh)
        ax.imshow(
            wsp, extent=[*xlim, *zlim], vmin=0, vmax=2, origin="lower", cmap="RdYlBu_r"
        )
    else:
        z = np.mean(sol.layout.z) if z is None else z
        _x, _y = np.linspace(*xlim, res), np.linspace(*ylim, res)
        xmesh, ymesh = np.meshgrid(_x, _y)
        wsp = sol.windfield.wsp(xmesh, ymesh, np.full_like(xmesh, z))
        ax.imshow(
            wsp, extent=[*xlim, *ylim], vmin=0, vmax=2, origin="lower", cmap="RdYlBu_r"
        )
        
    for (turb_x, turb_y, turb_z), rotor in zip(sol.layout, sol.rotors):
        # Draw turbine
        R = 0.5
        p = np.array([[0, 0], [+R, -R]])
        alpha = -rotor.tilt if xz_frame else rotor.yaw
        rotmat = np.array(
            [
                [np.cos(alpha), -np.sin(alpha)],
                [np.sin(alpha), np.cos(alpha)],
            ]
        )
        trans_vec = [[turb_x], [turb_z]] if xz_frame else [[turb_x], [turb_y]]
        p = rotmat @ p + np.array(trans_vec)

        ax.plot(p[0, :], p[1, :], "k", lw=2)
    ax.set(frame_on=frame)
    if axis is False:
        ax.set(xticks=[], yticks=[])
