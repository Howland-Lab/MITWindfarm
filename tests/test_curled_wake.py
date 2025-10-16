from pytest import fixture, mark, approx
import numpy as np
from mitwindfarm import Uniform, Layout
from mitwindfarm.windfarm import Windfarm, CurledWindfarm
from mitwindfarm.Rotor import UnifiedAD_TI, UnifiedAD
from UnifiedMomentumModel.Utilities.Geometry import calc_eff_yaw, eff_yaw_inv_rotation
from scipy.ndimage import rotate

def get_curled_windfarm(couple_x0 = False, TIamb = 0.05):
    base_windfield = Uniform(TIamb=TIamb)
    wf_curled = CurledWindfarm(
        rotor_model=UnifiedAD_TI(couple_x0 = couple_x0),
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

def get_gaussian_windfarm():
    base_windfield = Uniform()
    wf = Windfarm(
        rotor_model=UnifiedAD(),
        base_windfield=base_windfield,
    )
    return wf

def get_yaws_tilts(val):
    # create three sets of setpoints with same effective angle, one with just yaw,
    # one with just tilt, and one with yaw and tilt.
    yaw3 = np.radians(val)
    tilt3 = np.radians(val)
    eff_yaw = calc_eff_yaw(yaw3, tilt3)
    yaws, tilts = [eff_yaw, 0, yaw3], [0, eff_yaw, tilt3]
    return yaws, tilts



# test that wake is the same stamp rotated for different combinations of yaw and tilt
def test_wake_shape_rotation():
    wf_curled, layout = get_curled_windfarm()
    # get grid points
    pad, res = 2, 100
    ylim = (np.min(layout.y) - pad, np.max(layout.y) + pad)
    zlim = (np.min(layout.z) - pad, np.max(layout.z) + pad)
    _y, _z = np.linspace(*ylim, res), np.linspace(*zlim, res)
    ymesh, zmesh = np.meshgrid(_y, _z)
    # get solutions to all set point combos
    wsp_list = []
    ctprime = 1.33
    xval = 5 #D
    yaws, tilts = get_yaws_tilts(20)
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

def test_TI_rotor_equivalence():
    wf_curled_TI, layout = get_curled_windfarm(couple_x0 = False, TIamb = 0)
    wf_curled_TI_couple, _ = get_curled_windfarm(couple_x0 = True, TIamb = 0)
    wf = get_gaussian_windfarm()
    yaws, tilts = get_yaws_tilts(20)
    ctprime = 1.33
    for yaw, tilt in zip(yaws, tilts):
        setpoints = [(ctprime, yaw, tilt)]
        sol1 = wf_curled_TI(layout, setpoints)
        sol2 = wf_curled_TI_couple(layout, setpoints)
        sol3 = wf(layout, setpoints)
        # ensure that without TI, the new UMM models (with TI adjustments) are equivalent
        assert sol1.rotors[0].u4 ==  sol2.rotors[0].u4 ==  sol3.rotors[0].u4
        assert sol1.rotors[0].v4 ==  sol2.rotors[0].v4 ==  sol3.rotors[0].v4
        assert sol1.rotors[0].w4 ==  sol2.rotors[0].w4 ==  sol3.rotors[0].w4
        assert sol1.rotors[0].Cp ==  sol2.rotors[0].Cp ==  sol3.rotors[0].Cp

def test_TI_rotor_differences():
    wf_curled_TI, layout = get_curled_windfarm(couple_x0 = False, TIamb = 0.1)
    wf_curled_TI_couple, _ = get_curled_windfarm(couple_x0 = True, TIamb = 0.1)
    wf = get_gaussian_windfarm()
    yaws, tilts = get_yaws_tilts(20)
    ctprime = 1.33
    for yaw, tilt in zip(yaws, tilts):
        setpoints = [(ctprime, yaw, tilt)]
        sol1 = wf_curled_TI(layout, setpoints)
        sol2 = wf_curled_TI_couple(layout, setpoints)
        sol3 = wf(layout, setpoints)
        # ensure that without TI, the new UMM models (with TI adjustments) are equivalent
        assert sol1.rotors[0].extra.dp_NL != sol3.rotors[0].extra.dp_NL
        assert sol2.rotors[0].u4 != sol3.rotors[0].u4

test_wake_shape_rotation()
test_TI_rotor_equivalence()
test_TI_rotor_differences()
        