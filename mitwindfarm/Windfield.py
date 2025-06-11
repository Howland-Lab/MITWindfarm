"""
Windfield Abstraction and Concrete Windfield Implementation

This module defines an abstract base class, `Windfield`, representing a generic wind field,
and several concrete implementations, representing a various wind fields.

Classes:
- Windfield: Abstract base class for wind field models.
- Uniform: Concrete implementation of a uniform wind field.
- PowerLaw: Concrete implementation of a power law wind field.
- Superimposed:

Note that Superimposed is used internally, rather than as a user input.

Usage Example:
    wind_field = Uniform()  # Create a uniform wind field instance
    wind_speed = wind_field.wsp(x, y, z)  # Get wind speed at specified coordinates
    wind_direction = wind_field.wdir(x, y, z)  # Get wind direction at specified coordinates

Note: The methods wsp, TI, and wdir should be implemented in subclasses according to the specific wind field model.
"""

from abc import ABC, abstractmethod
from typing import Literal

from numpy.typing import ArrayLike
import numpy as np

from .Wake import Wake


class Windfield(ABC):
    """
    Abstract base class for wind field models.

    Subclasses must implement the wsp, TI, and wdir methods.
    """

    @abstractmethod
    def wsp(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        """
        Calculate wind speed at specified coordinates.

        Parameters:
        - x: x-coordinates.
        - y: y-coordinates.
        - z: z-coordinates.

        Returns:
        ArrayLike: Wind speed at the specified coordinates.
        """
        pass

    @abstractmethod
    def TI(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        """
        Calculate turbulence intensity at specified coordinates.

        Parameters:
        - x: x-coordinates.
        - y: y-coordinates.
        - z: z-coordinates.

        Returns:
        ArrayLike: Turbulence intensity at the specified coordinates.
        """
        pass

    @abstractmethod
    def wdir(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        """
        Calculate wind direction at specified coordinates.

        Parameters:
        - x: x-coordinates.
        - y: y-coordinates.
        - z: z-coordinates.

        Returns:
        ArrayLike: Wind direction at the specified coordinates.
        """
        pass


class Uniform(Windfield):
    """
    Concrete implementation of a uniform wind field.

    Methods:
    - wsp(x, y, z): Returns an array of U0 with the same shape as input coordinates.
    - TI(x, y, z): Returns an array of TIamb with the same shape as input coordinates
    - wdir(x, y, z): Returns an array of zeros with the same shape as input coordinates.
    """

    def __init__(self, U0: float = 1.0, TIamb: float = 0.0):
        self.U0 = U0
        self.TIamb = 0.0 if TIamb is None else TIamb

    def wsp(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        """
        Returns an array of value U0 with the same shape as input coordinates.

        Parameters:
        - x: x-coordinates.
        - y: y-coordinates.
        - z: z-coordinates.

        Returns:
        ArrayLike: Array of value U0 with the same shape as input coordinates.
        """
        return self.U0 * np.ones_like(x)

    def TI(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        """
        Calculate wind speed and turbulence intensity at specified coordinates.

        Parameters:
        - x: x-coordinates.
        - y: y-coordinates.
        - z: z-coordinates.

        Returns:
        ArrayLike: Turbulence intensity at the specified coordinates.
        """
        return self.TIamb * np.ones_like(x)

    def wdir(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        """
        Returns an array of zeros with the same shape as input coordinates.

        Parameters:
        - x: x-coordinates.
        - y: y-coordinates.
        - z: z-coordinates.

        Returns:
        ArrayLike: Array of zeros with the same shape as input coordinates.
        """
        return np.zeros_like(x)


class PowerLaw(Windfield):
    """
    Concrete implementation of a power law wind field.

    Methods:
    - shear(y): Returns the wind speed due to shear
    - wsp(x, y, z): Returns wind speed at a given height z
    - TI(x, y, z): Returns the input turbulence intensity TIamb with the same shape as input coordinates.
    - wdir(x, y, z): Returns an array of zeros with the same shape as input coordinates.
    """

    def __init__(self, Uref: float, zref: float, exp: float, TIamb: float = 0.0, veer: float = 0, ):
        self.Uref = Uref
        self.zref = zref
        self.exp = exp
        self.TIamb = TIamb
        self.veer = veer

    def shear(self, y):
        """
        Returns wind speed due to shear.
        """
        u = self.Uref * (y / self.zref) ** self.exp
        u = np.nan_to_num(u)
        return u

    def wsp(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        u = self.Uref * (z / self.zref) ** self.exp
        u = np.nan_to_num(u)
        return u

    def TI(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        return self.TIamb * np.ones_like(x)

    def wdir(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        """Linear direction shear with strength -veer (radians/length)"""
        return -self.veer * (z - self.zref)


class Superimposed(Windfield):
    """
    Concrete implementation of a superimposed wind field.

    Methods:
    - add_wake(base_windfield, wakes, method): Add provided wakes to the base windfield.
    - wsp(x, y, z): Returns wind speed at a given height z
    - TI(x, y, z): Returns the input turbulence intensity TIamb with the same shape as input coordinates.
    - wdir(x, y, z): Returns an array of zeros with the same shape as input coordinates.
    """

    def __init__(
        self,
        base_windfield: Windfield,
        wakes: list[Wake],
        method=Literal["linear", "quadratic", "dominant", "niayifar"],
    ):
        self.base_windfield = base_windfield
        self.wakes = wakes
        self.method = method

    def add_wake(self, wake: Wake):
        self.wakes.append(wake)

    def wsp(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        """
        Returns an array of wind speed based on the windfield's wakes.

        Parameters:
        - x: x-coordinates.
        - y: y-coordinates.
        - z: z-coordinates.

        Returns:
        ArrayLike: Array of ind speed based on the windfield's wakes with the same shape as input coordinates.
        """
        wsp_base = self.base_windfield.wsp(x, y, z)
        deficits = []
        for wake in self.wakes:
            if self.method == "niayifar":
                deficits.append(wake.niayifar_deficit(x, y, z))
            else:
                deficits.append(wake.deficit(x, y, z))

        if len(deficits) == 0:
            deficits.append(np.zeros_like(wsp_base))

        if (self.method == "linear") | (self.method == "niayifar"):
            wsp_out = wsp_base - np.sum(deficits, axis=0)
        elif self.method == "quadratic":
            wsp_out = wsp_base - np.sqrt(np.sum(np.array(deficits) ** 2, axis=0))
        elif self.method == "dominant":
            wsp_out = wsp_base - np.array(deficits).max(axis=0, initial=0)
        else:
            raise NotImplementedError

        return wsp_out

    def TI(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        """
        Returns the turbulence intensity of the wake with the most added turbulence.
        """
        TI_base = self.base_windfield.TI(x, y, z)

        max_WATI = np.zeros_like(TI_base)
        for wake in self.wakes:
            max_WATI = np.maximum(wake.wake_added_turbulence(x, y, z), max_WATI)

        TI_out = np.sqrt(TI_base**2 + max_WATI**2)

        return TI_out

    def wdir(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        """
        Returns an array of zeros with the same shape as input coordinates.

        Parameters:
        - x: x-coordinates.
        - y: y-coordinates.
        - z: z-coordinates.

        Returns:
        ArrayLike: Array of zeros with the same shape as input coordinates.
        """
        return self.base_windfield.wdir(x, y, z)


class LogWindfield(Windfield):
    """
    Logarithmic layer wind field taking input surface roughness
    """

    def __init__(
        self,
        U0: float = 1,
        z0: float = None,
        TIamb: float = 0,
        kappa: float = 0.4,
        veer: float = 0,
        zref: float = None,
    ):
        """
        Initialize the wind field with specified parameters.

        Parameters:
        - U0: Reference wind speed (m/s or non-dim).
            If zref is specified, then U0 is the wind speed at zref.
            If zref is not specified, then U0 is the friction velocity u_*.
        - z0: Surface roughness length (m or non-dim).
        - TIamb: Ambient turbulence intensity (unitless).
        - kappa: von Karman constant (default is 0.4).
        - veer: Wind direction veer (radians/length, default is 0).
            Note: positive wind veer is negative above zref
        - zref, optional: Reference height for wind profile (default is None).
        """
        self.U0 = U0
        self.z0 = z0
        self.TIamb = TIamb
        self.kappa = kappa
        self.veer = veer
        if zref is not None:
            self.zref = zref
            self.ustar = U0 * kappa / np.log(zref / z0)
        elif veer:
            raise ValueError("zref must be specified if veer is non-zero.")
        else:
            self.ustar = self.U0

    def wsp(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        _z = np.clip(z / self.z0, 1, None)  # avoid negative velocities
        return self.ustar / self.kappa * np.log(_z)

    def TI(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        return self.TIamb * np.ones_like(z)

    def wdir(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        return -self.veer * (z - self.zref)


class ArbitraryZWindfield(Windfield):
    """
    Aribtrary wind field in the z-direction with no dependence on x or y

    Methods:
    - wsp(x, y, z): Returns an array of U0 with the same shape as input coordinates.
    - TI(x, y, z): Returns an array of TIamb with the same shape as input coordinates
    - wdir(x, y, z): Returns an array of zeros with the same shape as input coordinates.
    """

    def __init__(self, z, U_z=None, wdir_z=None, TIamb_z=None, TIamb=0):
        self.z = z
        self.U_z = U_z  # required
        self.wdir_z = wdir_z  # optional
        self.TIamb_z = TIamb_z  # optional
        self.TIamb = TIamb  # optional, constant TI value

    def wsp(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        return np.interp(z, self.z, self.U_z)

    def TI(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        if self.TIamb_z is not None:
            return np.interp(z, self.z, self.TIamb_z)
        else:
            return self.TIamb * np.ones_like(z)

    def wdir(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        if self.wdir_z is not None:
            return np.interp(z, self.z, self.wdir_z)
        else:
            return np.zeros_like(z)


class ShearedWindfield(Windfield):
    """
    Windfield with linear gradients of U and alpha but constant U in veer
    """

    def __init__(
        self,
        U0: float = 1,
        dUdz: float = 0,
        dalphadz: float = 0,
        TIamb: float = 0,
        clip_wsp=[0.1, None],
        clip_wdir=[-np.pi * 0.45, np.pi * 0.45],
    ):
        self.U0 = U0
        self.dUdz = dUdz
        self.dalphadz = dalphadz
        self.TIamb = TIamb
        self.clip_wsp = clip_wsp
        self.clip_wdir = clip_wdir

    def wsp(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        return np.clip(self.U0 * (1 + self.dUdz * z), *self.clip_wsp)

    def TI(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        return self.TIamb * np.ones_like(z)

    def wdir(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        return np.clip(self.dalphadz * z, *self.clip_wdir)


class UVShearedWindfield(Windfield):
    """
    Windfield with linear gradients of u, v instead of U, alpha
    """

    def __init__(
        self, U0: float = 1, dudz: float = 0, dvdz: float = 0, TIamb: float = 0
    ):
        self.U0 = U0
        self.dudz = dudz
        self.dvdz = dvdz
        self.TIamb = TIamb

    def wsp(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        u = self.U0 * (1 + self.dudz * z)
        v = self.U0 * self.dvdz * z
        return np.sqrt(u**2 + v**2)

    def TI(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        return self.TIamb * np.ones_like(z)

    def wdir(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        u = self.U0 * (1 + self.dudz * z)
        v = self.U0 * self.dvdz * z
        return np.arctan2(v, u)  # in radians
