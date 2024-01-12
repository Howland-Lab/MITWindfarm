import numpy as np
from mitwindfarm.Windfield import Windfield, Uniform


def test_uniform_windfield_wsp():
    wind_field = Uniform()
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    z = np.array([7, 8, 9])

    wind_speed = wind_field.wsp(x, y, z)

    assert np.array_equal(wind_speed, np.ones_like(x))


def test_uniform_windfield_wdir():
    wind_field = Uniform()
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    z = np.array([7, 8, 9])

    wind_direction = wind_field.wdir(x, y, z)

    assert np.array_equal(wind_direction, np.zeros_like(x))


def test_custom_windfield():
    class CustomWindfield(Windfield):
        def wsp(self, x, y, z):
            return x + y + z

        def wdir(self, x, y, z):
            return x * y * z

    custom_windfield = CustomWindfield()
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    z = np.array([7, 8, 9])

    wind_speed = custom_windfield.wsp(x, y, z)
    wind_direction = custom_windfield.wdir(x, y, z)

    assert np.array_equal(wind_speed, x + y + z)
    assert np.array_equal(wind_direction, x * y * z)
