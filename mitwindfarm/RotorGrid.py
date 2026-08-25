from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike


class RotorGrid(ABC):
    """
    Abstract base class for rotor grids.

    Subclasses must implement the `grid_points` and `average` methods.
    """
    @abstractmethod
    def grid_points(self) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Return the grid points to be sampled given turbine location.
        """
        ...

    @abstractmethod
    def average(self, value: ArrayLike) -> float:
        """
        Return the numerically integrated wind speeds sampled at grid point locations.
        """
        ...


class Point:
    def grid_points(self) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Returns the a single point (0, 0, 0) at the rotor center using a `Point` grid.

        Args: None

        Returns:
            (ndarray, ndarray, ndarray): X, Y, Z coordinates of grid points -  in this case the origin.
        """
        return np.array([0.0]), np.array([0.0]), np.array([0.0])

    def average(self, U: ArrayLike) -> ArrayLike:
        """
        Numerically integrate wind speeds sampled at grid point locations
        defined by Point.grid_points.

        Args:
            U (ArrayLike): Streamwise wind speeds.

        Returns:
            ArrayLike: Array of Rotor-averaged wind speeds for each rotor.
            In this case, just a mean value of the argument U.
        """
        return np.mean(U)


class Line:
    def __init__(self, disc=100):
        # predefine cartesian grid for performing REWS
        self.disc = disc
        self.xs = np.zeros(disc)
        self.ys = np.linspace(-0.5, 0.5, disc)
        self.zs = np.zeros(disc)

    def grid_points(self) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Returns the grid points to be sampled given the rotor location using a `Line` grid.

        Args: None.

        Returns:
            (ndarray, ndarray, ndarray): X, Y, Z coordinates of grid points.
        """
        return self.xs, self.ys, self.zs

    def average(self, U: ArrayLike) -> ArrayLike:
        """
        Numerically integrate wind speeds sampled at grid point locations
        defined by Line.grid_points.
        TODO: this doesn't seem to be accurate to code below - doesn't use Line.gridpoints

        Args:
            U (ArrayLike): Streamwise wind speeds.

        Returns:
            ArrayLike: Array of Rotor-averaged wind speeds for each rotor.
            In this case, just a mean value of the argument U.
        """

        return np.mean(U)


class Area:
    def __init__(self, r_disc=10, theta_disc=10):
        # predefine polar grid for performing REWS
        self.r_disc, self.theta_disc = r_disc, theta_disc
        rs = np.linspace(0, 0.5, r_disc)
        self.thetas = np.linspace(0, 2 * np.pi, theta_disc)

        self.r_mesh, self.theta_mesh = np.meshgrid(rs, self.thetas)

        self.xs = np.zeros_like(self.r_mesh)
        self.ys = self.r_mesh * np.sin(self.theta_mesh)
        self.zs = self.r_mesh * np.cos(self.theta_mesh)

    def grid_points(self) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Returns the grid points to be sampled given the rotor locations using a `Area` grid.

        Args: None.

        Returns:
            (ndarray, ndarray, ndarray): X, Y, Z coordinates of grid points within the rotor.
        """

        return self.xs, self.ys, self.zs

    def average(self, U: ArrayLike) -> ArrayLike:
        """
        Numerically integrate wind speeds sampled at grid point locations
        defined by Point.grid_points.

        Args:
            U (ArrayLike): Streamwise wind speeds.

        Returns:
            ArrayLike: Array of Rotor-averaged wind speeds for each rotor.
        """

        return (
            4
            / np.pi
            * np.trapezoid(
                np.trapezoid(self.r_mesh * U, self.r_mesh, axis=-1), self.thetas, axis=-1
            )
        )
