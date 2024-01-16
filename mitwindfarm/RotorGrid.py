from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike


class RotorGrid(ABC):
    @abstractmethod
    def grid_points(self) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        ...

    @abstractmethod
    def average(self, value: ArrayLike) -> float:
        ...


class Point:
    def grid_points(self) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Returns the grid points to be sampled given the turbine locations, X_t,
        Y_t

        Args:
            X_t (ndarray): Streamwise locations of turbines.
            Y_t (ndarray): Lateral locations of turbines.

        Returns:
            (ndarray, ndarray, ndarray): X, Y, Z coordinates of grid points.
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
        """
        return np.mean(U)


class Line:
    def __init__(self, disc=100):
        # predefine polar grid for performing REWS
        self.disc = disc
        self.xs = np.zeros(disc)
        self.ys = np.linspace(-0.5, 0.5, disc)
        self.zs = np.zeros(disc)

    def grid_points(self) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Returns the grid points to be sampled given the turbine locations, X_t,
        Y_t

        Args:
            X_t (List[float]): Streamwise locations of turbines.
            Y_t (List[float]): Lateral locations of turbines.

        Returns:
            (ndarray, ndarray, ndarray): X, Y, Z coordinates of grid points.
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
        Returns the grid points to be sampled given the turbine locations, X_t,
        Y_t

        Args:
            X_t (List[float]): Streamwise locations of turbines.
            Y_t (List[float]): Lateral locations of turbines.

        Returns:
            (ndarray, ndarray, ndarray): X, Y, Z coordinates of grid points.
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
            * np.trapz(
                np.trapz(self.r_mesh * U, self.r_mesh, axis=-1), self.thetas, axis=-1
            )
        )
