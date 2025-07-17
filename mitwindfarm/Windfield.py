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
            if (self.method == "niayifar") | (self.method == "nquadratic"):
                deficits.append(wake.niayifar_deficit(x, y, z))
            else:
                deficits.append(wake.deficit(x, y, z))
    
        if len(deficits) == 0:
            deficits.append(np.zeros_like(wsp_base))

        if (self.method == "linear") | (self.method == "niayifar"):
            wsp_out = wsp_base - np.sum(deficits, axis=0)
        elif (self.method == "quadratic") | (self.method == "nquadratic"):
            wsp_out = wsp_base - np.sqrt(np.sum(np.array(deficits)**2, axis=0))
        elif self.method == "dominant":
            wsp_out = wsp_base - np.array(deficits).max(axis=0, initial=0)
        else:
            raise NotImplementedError

        return wsp_out
    

    def mem_eff_wsp(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        """
        Returns an array of wind speed based on superposing the windfield's wakes. Runs without creating a
        x by y by z by n_rotors array to conserve memory. 

        Parameters:
        - x: x-coordinates.
        - y: y-coordinates.
        - z: z-coordinates.

        Returns:
        ArrayLike: Array of ind speed based on the windfield's wakes with the same shape as input coordinates.
        """    
        wsp_base = self.base_windfield.wsp(x, y, z)
        tot_deficit = np.zeros_like(wsp_base)
        deficit_count = 0

        for wake in self.wakes:
            if (self.method == "niayifar") | (self.method == "nquadratic"):
                deficit = wake.niayifar_deficit(x, y, z)
            else: 
                deficit = wake.deficit(x, y, z)

            if deficit is not None:
                deficit_count += 1

            # Compute linear methods by first totalling deficit fields
            if (self.method == "linear") | (self.method == "niayifar"):
                tot_deficit += deficit
            # Compute quadratic methods by first summing squares of deficit fields
            elif (self.method == "quadratic") | (self.method == "nquadratic"):
                tot_deficit += deficit**2
            # Compute dominant method by first checking maximum of deficit fields
            elif self.method == "dominant":
                # Stack current deficit maximum and deficit
                deficits = np.stack((tot_deficit, deficit), axis = 0)
                # Store maximum of the two
                tot_deficit = deficits.max(axis = 0, initial = 0)
            else:
                raise NotImplementedError
            
        # Handle no deficits
        if deficit_count == 0:
            wsp_out = wsp_base
        # Output linear and dominant methods by subtracting total of deficits from base
        elif (self.method == "linear") | (self.method == "niayifar") | (self.method == "dominant"):
            wsp_out = wsp_base - tot_deficit
        # Output quadratic methods by subtracting square root of tot_deficit from base
        elif (self.method == "quadratic") | (self.method == "nquadratic"):
            wsp_out = wsp_base - np.sqrt(tot_deficit)

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
        return np.zeros_like(x)
