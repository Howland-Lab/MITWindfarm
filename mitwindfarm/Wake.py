from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import cumtrapz
from scipy.special import erf

if TYPE_CHECKING:
    from .Rotor import RotorSolution


class Wake(ABC):
    @abstractmethod
    def __init__(
        self, x: float, y: float, z: float, rotor_sol: "RotorSolution", **kwargs
    ):
        ...

    @abstractmethod
    def deficit(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        ...

    @abstractmethod
    def centerline(self, x: ArrayLike) -> ArrayLike:
        ...


class WakeModel(ABC):
    @abstractmethod
    def __call__(self, x, y, z, rotor_sol: "RotorSolution") -> Wake:
        ...


class GaussianWake(Wake):
    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        rotor_sol: "RotorSolution",
        sigma=0.25,
        kw=0.07,
    ):
        self.x, self.y, self.z = x, y, z
        self.rotor_sol = rotor_sol
        self.sigma, self.kw = sigma, kw

    def centerline(self, x_glob: ArrayLike, dx=0.05) -> ArrayLike:
        """
        Solves Eq. C4. Returns centerline y position in global coordinates.
        """
        x = x_glob - self.x
        xmax = np.max(x)
        _x = np.arange(0, max(xmax, 2 * dx), dx)
        d = self._wake_diameter(_x)

        dv = -0.5 / d**2 * (1 + erf(_x / (np.sqrt(2) / 2)))
        _yc_temp = cumtrapz(-dv, dx=dx, initial=0)

        yc_temp = np.interp(x, _x, _yc_temp, left=0)

        return yc_temp * self.rotor_sol.v4 + self.y

    def _wake_diameter(self, x: ArrayLike) -> ArrayLike:
        """
        Solves the normalized far-wake diameter (between C1 and C2)
        """
        return 1 + self.kw * np.log(1 + np.exp(2 * (x - 1)))

    def _du(self, x: ArrayLike, wake_diameter: Optional[float] = None) -> ArrayLike:
        """
        Solves Eq. C2
        """
        d = self._wake_diameter(x) if wake_diameter is None else wake_diameter

        du = 0.5 * (1 - self.rotor_sol.u4) / d**2 * (1 + erf(x / (np.sqrt(2) / 2)))
        return du

    def deficit(self, x_glob: ArrayLike, y_glob: ArrayLike, z_glob=0) -> ArrayLike:
        """
        Solves Eq. C1
        """
        x, y, z = x_glob - self.x, y_glob - self.y, z_glob - self.z
        d = self._wake_diameter(x)
        yc = self.centerline(x_glob) - self.y
        du = self._du(x, wake_diameter=d)
        deficit_ = (
            1
            / (8 * self.sigma**2)
            * np.exp(-(((y - yc) ** 2 + z**2) / (2 * self.sigma**2 * d**2)))
        )

        return deficit_ * du

    def line_deficit(self, x: np.array, y: np.array):
        """
        Returns the deficit at hub height averaged along a lateral line of
        length 1, centered at (x, y).
        """

        d = self._wake_diameter(x)
        yc = self.centerline(x)
        du = self._du(x, wake_diameter=d)

        erf_plus = erf((y + 0.5 - yc) / (np.sqrt(2) * self.sigma * d))
        erf_minus = erf((y - 0.5 - yc) / (np.sqrt(2) * self.sigma * d))

        deficit_ = np.sqrt(2 * np.pi) * d / (16 * self.sigma) * (erf_plus - erf_minus)

        return deficit_ * du


class GaussianWakeModel(WakeModel):
    def __init__(self, sigma=0.25, kw=0.07):
        self.sigma = sigma
        self.kw = kw

    def __call__(self, x, y, z, rotor_sol: "RotorSolution") -> GaussianWake:
        return GaussianWake(x, y, z, rotor_sol, sigma=self.sigma, kw=self.kw)
