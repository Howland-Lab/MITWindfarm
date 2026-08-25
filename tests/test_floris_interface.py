import numpy as np

from floris import FlorisModel

from mitwindfarm import FlorisCurledWindfarm
import pytest

LAYOUT_X = [0.0, 1200.0, 2400.0]
LAYOUT_Y = [0.0, 0.0, 0.0]
WIND_SPEEDS = [9.0, 9.0]
WIND_DIRECTIONS = [270.0, 180.0]
TURBULENCE_INTENSITIES = [0.06, 0.06]

def _setup_basic_floris_model():
    fmodel = FlorisModel("defaults")
    fmodel.set(
        layout_x=LAYOUT_X,
        layout_y=LAYOUT_Y,
        wind_speeds=WIND_SPEEDS,
        wind_directions=WIND_DIRECTIONS,
        turbulence_intensities=TURBULENCE_INTENSITIES,
        turbine_type=["iea_15MW"] * len(LAYOUT_X),
    )
    return fmodel

def test_set_wake_model():
    fmodel = _setup_basic_floris_model()
    fmodel.run()
    powers_base = fmodel.get_turbine_powers()

    fmodel.set_wake_model(FlorisCurledWindfarm())
    fmodel.run()
    powers_test = fmodel.get_turbine_powers()

    # Confirm shapes match, freestream powers match, powers in wake differ
    assert powers_base.shape == powers_test.shape
    assert np.isclose(powers_base[0, 0], powers_test[0, 0])
    assert not np.any(np.isclose(powers_base[0,1:], powers_test[0,1:]))
    assert np.allclose(powers_test[1,0], powers_test[1,1:])  # All turbines unwaked

def test_above_rated():
    fmodel = _setup_basic_floris_model()
    fmodel.set_wake_model(FlorisCurledWindfarm())
    fmodel.run()
    powers_below_rated = fmodel.get_turbine_powers()

    # Increase wind speed to above rated
    fmodel.set(
        wind_speeds=[15.0, 15.0],
    )
    fmodel.run()
    powers_above_rated = fmodel.get_turbine_powers()

    # Confirm shapes match, freestream powers match, powers in wake differ
    assert powers_below_rated.shape == powers_above_rated.shape
    assert np.all(powers_above_rated > powers_below_rated)
    assert not np.allclose(powers_below_rated[0,:], powers_above_rated[0,:])

def test_yawed():
    fmodel = _setup_basic_floris_model()
    fmodel.set_wake_model(FlorisCurledWindfarm())
    fmodel.run()
    powers_no_yaw = fmodel.get_turbine_powers()

    # Increase wind speed to above rated
    fmodel.set(
        yaw_angles=[[30.0, 30.0, 0.0], [30.0, 20.0, 0.0]],
    )
    fmodel.run()
    powers_with_yaw = fmodel.get_turbine_powers()

    # Check that freestream turbine yawed produced less power
    assert powers_no_yaw[0,0] > powers_with_yaw[0,0]
    # Check that freestream turbine yawed produces same power in both conditions
    assert np.isclose(powers_with_yaw[0,0], powers_with_yaw[1,0])
    # Check that downstream turbines produce more power with wake steering
    assert powers_with_yaw[0,1] > powers_no_yaw[0,1]

def test_user_flags():
    fmodel = _setup_basic_floris_model()

    # Tilt flag behavior
    fmodel.set_wake_model(FlorisCurledWindfarm(use_floris_tilt=True))
    fmodel.run()
    powers_tilted = fmodel.get_turbine_powers()
    fmodel.set_wake_model(FlorisCurledWindfarm(use_floris_tilt=False))
    fmodel.run()
    powers_not_tilted = fmodel.get_turbine_powers()

    # Freestream turbines produce more power when not tilted
    assert powers_tilted[0,0] < powers_not_tilted[0,0]

def test_raises_errors():
    fmodel = _setup_basic_floris_model()
    fmodel.set_wake_model(FlorisCurledWindfarm())

    # Set non-uniform turbine types
    fmodel.set(turbine_type=["iea_15MW", "iea_15MW", "nrel_5MW"])
    with pytest.raises(NotImplementedError, match="Varying turbine models not supported"):
        fmodel.run()

    # Reset turbine types; apply heterogeneous wind speeds
    fmodel.set(turbine_type=["iea_15MW"]*3)
    fmodel.run() # Make sure runs through with uniform turbine types

    heterogeneous_inflow_config = {
        "speed_multipliers": [[1.0, 1.25, 1.0, 1.25], [1.0, 1.25, 1.0, 1.25]],
        "x": [-500.0, -500.0, 1000.0, 1000.0],
        "y": [-500.0, 1000.0, -500.0, 1000.0],
    }
    fmodel.set(heterogeneous_inflow_config=heterogeneous_inflow_config)
    with pytest.raises(NotImplementedError, match="Heterogeneous inflows are not supported"):
        fmodel.run()
