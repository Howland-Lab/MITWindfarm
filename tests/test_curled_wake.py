from pytest import fixture, mark, approx
import numpy as np
from mitwindfarm import Uniform, Layout
from mitwindfarm.windfarm import Windfarm, CurledWindfarm
from mitwindfarm.Rotor import UnifiedAD_TI
from UnifiedMomentumModel.Utilities.Geometry import calc_eff_yaw, eff_yaw_inv_rotation
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

def get_curled_windfarm():
    base_windfield = Uniform(TIamb=0.05)  # 5% ambient TI

    wf_curled = CurledWindfarm(
        rotor_model=UnifiedAD_TI(),
        base_windfield=base_windfield,
        solver_kwargs=dict(
            dy=1/10,
            dz=1/10,
            integrator="scipy_rk23",  # see mitwindfarm.utils.integrate
            k_model="k-l",  # alternatives: "const", "2021"
            verbose=False,
        ),
    )

    xturb, yturb, zturb = [0], [0], [0]
    layout = Layout(xturb, yturb, zturb)
    return wf_curled, layout


# test that wake is the same stamp rotated for different combinations of yaw and tilt
def test_wake_shape_rotation():
    # create three sets of setpoints with same effective angle, one with just yaw,
    # one with just tilt, and one with yaw and tilt.
    yaw3 = np.radians(20)
    tilt3 = np.radians(20)
    eff_yaw = calc_eff_yaw(yaw3, tilt3)
    print(np.rad2deg(eff_yaw))
    yaws, tilts = [eff_yaw, 0, yaw3], [0, eff_yaw, tilt3]
    wf_curled, layout = get_curled_windfarm()
    # get grid points
    pad, res = 2, 100
    ylim = (np.min(layout.y) - pad, np.max(layout.y) + pad)
    zlim = (np.min(layout.z) - pad, np.max(layout.z) + pad)
    _y, _z = np.linspace(*ylim, res), np.linspace(*zlim, res)
    ymesh, zmesh = np.meshgrid(_y, _z)
    # get solutions to all set point combos
    wsp_list = []
    ctprime = 2
    xval = 5 #D
    for yaw, tilt in zip(yaws, tilts):
        setpoints = [(ctprime, yaw, tilt)]
        sol = wf_curled(layout, setpoints)
        wsp = sol.windfield.wsp(np.full_like(ymesh, xval), ymesh, zmesh)
        wsp_list.append(wsp)

    # rotate solutions to be in "yaw-only" frame
    rotated_yaw = rotate(wsp_list[0], 0, reshape=False, order=1, mode='constant', cval=1)
    rotated_tilt = rotate(wsp_list[1], -90, reshape=False, order=1, mode='constant', cval=1)
    rotated_yaw_and_tilt = rotate(wsp_list[2], -45, reshape=False, order=1, mode='constant', cval=1)
    
    # check that there is less than 5% difference between each rotated dataset compared to yaw simulation
    assert(np.allclose(rotated_yaw, wsp_list[0], atol = 0.05 * np.min(wsp_list[0])))
    assert(np.allclose(rotated_tilt, wsp_list[0], atol = 0.05 * np.min(wsp_list[0])))
    assert(np.allclose(rotated_yaw_and_tilt, wsp_list[0], atol = 0.05 * np.min(wsp_list[0])))

test_wake_shape_rotation()   
        