from pytest import fixture, mark, approx
import numpy as np
from mitwindfarm import RotorSolution, GaussianWakeModel

@fixture
def grid_info():
    xturb = 1
    npoints = 20
    _x, _y, _z = np.linspace(-1, 10, npoints), np.linspace(-3, 3, npoints), np.linspace(-3, 3, npoints)
    dx, dy, dz = np.abs(_x[0] - _x[1]), np.abs(_y[0] - _y[1]), np.abs(_z[0] - _z[1])
    return xturb, _x, _y, _z, dx, dy, dz

@fixture
def wake_model():
    return GaussianWakeModel()

@fixture
def wake_model_args():
    yaw, tilt = 0, 0
    Cp, Ct, Ctprime = 0, 0, 0
    an, u4, REWS = 0.3, 0.5, 1.0
    return yaw, tilt, Cp, Ct, Ctprime, an, u4, REWS


@mark.parametrize('yturb, zturb, v4, w4', [(1, 1, 0, 0), (1, 0, 0.5, 0.0), (0, 1, -0.5, 0.0), (-1, 1, 0, 0.5), (1, -1, 0, -0.5), (-1, -1, 0.5, 0.5), (0, 0, -0.5, 0.5)])
def test_v4_w4_centerline_wake(grid_info, wake_model, wake_model_args, yturb, zturb, v4, w4):
    print("A")
    print(yturb, zturb)
    # get data from fixtures
    xturb, _x, _y, _z, dx, dy, dz = grid_info
    yaw, tilt, Cp, Ct, Ctprime, an, u4, REWS = wake_model_args
    # create rotor solution and wake
    rotor_sol = RotorSolution(yaw, tilt, Cp, Ct, Ctprime, an, u4, v4, w4, REWS)
    wake = wake_model(xturb, yturb, zturb, rotor_sol)
    y_centerline, z_centerline = wake.centerline(_x)
    # assert that the centerline goes in the same direction as the v4/w4 velocity
    assert all(y_centerline >= yturb) if wake.rotor_sol.v4 > 0 else all(y_centerline <= yturb)
    assert all(z_centerline >= zturb) if wake.rotor_sol.w4 > 0 else all(z_centerline <= zturb)
    # find grid indices of centerline values
    yc_idxs = np.array([np.abs(_y - yc).argmin() for yc in y_centerline])
    zc_idxs = np.array([np.abs(_z - zc).argmin() for zc in z_centerline])
    # generate deficits
    xy_meshes = np.meshgrid(_x, _y)
    xy_deficit = wake.deficit(*xy_meshes, zturb) # x-y plane
    xz_meshes = np.meshgrid(_x, _z)
    xz_deficit = wake.deficit(xz_meshes[0], yturb, xz_meshes[1]) # x-z plane 
    yz_meshes = np.meshgrid(_y, _z)
    yz_deficit = wake.deficit(xturb, *yz_meshes) # y-z plane
    # assert that the y/z deficits matches the centerlines
    max_wake_yidx = np.argmax(xy_deficit, axis=0)
    max_wake_zidx = np.argmax(xz_deficit, axis=0)
    assert all(max_wake_yidx == yc_idxs)
    assert all(max_wake_zidx == zc_idxs)
    # assert that the wake is centered on tubine in y and z
    wake_center_zidx, wake_center_yidx = np.unravel_index(np.argmax(yz_deficit), yz_deficit.shape)
    assert approx(_y[wake_center_yidx], abs = dy) == yturb
    assert approx(_z[wake_center_zidx], abs = dz) == zturb
    return