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

from numpy.typing import ArrayLike
import numpy as np


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
        return np.ones_like(x)

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
