"""
Windfield Abstraction and Uniform Windfield Implementation

This module defines an abstract base class, `Windfield`, representing a generic wind field,
and a concrete implementation, `Uniform`, representing a uniform wind field.

Classes:
- Windfield: Abstract base class for wind field models.
- Uniform: Concrete implementation of a uniform wind field.

Usage Example:
    wind_field = Uniform()  # Create a uniform wind field instance
    wind_speed = wind_field.wsp(x, y, z)  # Get wind speed at specified coordinates
    wind_direction = wind_field.wdir(x, y, z)  # Get wind direction at specified coordinates

Note: The methods wsp and wdir should be implemented in subclasses according to the specific wind field model.
"""

from abc import ABC, abstractmethod
from typing import Literal

from numpy.typing import ArrayLike
import numpy as np

from .Wake import Wake


class Windfield(ABC):
    """
    Abstract base class for wind field models.

    Subclasses must implement the wsp and wdir methods.
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
        ArrayLike: Wind speed at the specified coordinates.
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
    - wsp(x, y, z): Returns an array of ones with the same shape as input coordinates.
    - wdir(x, y, z): Returns an array of zeros with the same shape as input coordinates.
    """

    def __init__(self, U0: float = 1.0, TIamb: float = 0.0):
        self.U0 = U0
        self.TIamb = 0.0 if TIamb is None else TIamb

    def wsp(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        """
        Returns an array of ones with the same shape as input coordinates.

        Parameters:
        - x: x-coordinates.
        - y: y-coordinates.
        - z: z-coordinates.

        Returns:
        ArrayLike: Array of ones with the same shape as input coordinates.
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
        ArrayLike: Wind speed at the specified coordinates.
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
    def __init__(self, Uref: float, zref: float, exp: float, TIamb: float = 0.0):
        self.Uref = Uref
        self.zref = zref
        self.exp = exp
        self.TIamb = TIamb

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
        return np.zeros_like(x)


class Superimposed(Windfield):
    def __init__(
        self,
        base_windfield: Windfield,
        wakes: list[Wake],
        method=Literal["linear", "quadratic", "dominant"],
    ):
        self.base_windfield = base_windfield
        self.wakes = wakes
        self.method = method

    def add_wake(self, wake: Wake):
        self.wakes.append(wake)

    def wsp(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        """
        Returns an array of ones with the same shape as input coordinates.

        Parameters:
        - x: x-coordinates.
        - y: y-coordinates.
        - z: z-coordinates.

        Returns:
        ArrayLike: Array of ones with the same shape as input coordinates.
        """
        wsp_base = self.base_windfield.wsp(x, y, z)
        deficits = []
        for wake in self.wakes:
            deficits.append(wake.deficit(x, y, z))

        if len(deficits) == 0:
            deficits.append(np.zeros_like(wsp_base))

        if self.method == "linear":
            wsp_out = wsp_base - np.sum(deficits, axis=0)
        elif self.method == "quadratic":
            wsp_out = wsp_base - np.sqrt(np.sum(np.array(deficits)**2, axis=0))
        elif self.method == "dominant":
            wsp_out = wsp_base - np.array(deficits).max(axis=0, initial=0)
        else:
            raise NotImplementedError

        return wsp_out

    def TI(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
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
        return np.zeros_like(x)
