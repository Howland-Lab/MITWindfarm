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
    ): ...

    @abstractmethod
    def deficit(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def wake_added_turbulence(
        self, x: ArrayLike, y: ArrayLike, z: ArrayLike
    ) -> ArrayLike: ...

    @abstractmethod
    def deficit_and_WATI(
        self, x: ArrayLike, y: ArrayLike, z: ArrayLike
    ) -> ArrayLike: ...

    @abstractmethod
    def centerline(self, x: ArrayLike) -> ArrayLike: ...


class WakeModel(ABC):
    @abstractmethod
    def __call__(self, x, y, z, rotor_sol: "RotorSolution") -> Wake: ...


class GaussianWake(Wake):
    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        rotor_sol: "RotorSolution",
        sigma: float = 0.25,
        kw: float = 0.07,
        TIamb: float = None,
        xmax: float = 100.0,
        dx: float = 0.05,
    ):
        self.x, self.y, self.z = x, y, z
        self.rotor_sol = rotor_sol
        self.sigma, self.kw = sigma, kw
        self.TIamb = TIamb or 0.0

        # precompute centerline far downstream
        self.x_centerline, self.y_centerline = self._centerline(xmax, dx)

    def __repr__(self):
        return f"GaussianWake(x={self.x}, y={self.y}, z={self.z}, sigma={self.sigma}, kw={self.kw})"

    def _centerline(self, xmax: float, dx: float = 0.05) -> ArrayLike:
        """
        Solves Eq. C4. Returns centerline y position in global coordinates.
        """

        _x = np.arange(0, max(xmax, 2 * dx), dx)
        d = self._wake_diameter(_x)

        dv = -0.5 / d**2 * (1 + erf(_x / (np.sqrt(2) / 2)))
        _yc = cumtrapz(-dv, dx=dx, initial=0)

        return _x, _yc

    def centerline(self, x_glob: ArrayLike) -> ArrayLike:
        """
        Solves Eq. C4. Returns centerline y position in global coordinates.
        """
        x = x_glob - self.x

        yc_temp = np.interp(x, self.x_centerline, self.y_centerline, left=0)

        return yc_temp * self.rotor_sol.v4 + self.y

    def centerline_wake_added_turb(self, x: ArrayLike) -> ArrayLike:
        """
        Returns the centerline wake-added turbulence intensity (WATI) based on
        the model by Crespo and Hernandez (1996).
        """
        if self.TIamb is None or self.TIamb == 0.0:
            return np.zeros_like(x)

        else:
            x = x
            with np.errstate(all="ignore"):
                WATI = (
                    0.73
                    * (self.rotor_sol.an / self.rotor_sol.REWS) ** 0.8325
                    * self.TIamb ** (-0.0325)
                    * np.maximum(x, 0.1) ** (-0.32)
                )
            WATI[x < 0.1] = 0.0
            return WATI

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
        gaussian_ = (
            1
            / (8 * self.sigma**2)
            * np.exp(-(((y - yc) ** 2 + z**2) / (2 * self.sigma**2 * d**2)))
        )

        return gaussian_ * du

    def wake_added_turbulence(
        self, x_glob: ArrayLike, y_glob: ArrayLike, z_glob=0
    ) -> ArrayLike:
        """
        Returns wake added turbulence intensity caused by a wake at particular
        points in space. Laterally smeared with the same gaussian as the wake
        deficit model.
        """
        x, y, z = x_glob - self.x, y_glob - self.y, z_glob - self.z
        d = self._wake_diameter(x)
        yc = self.centerline(x_glob) - self.y
        WATI = self.centerline_wake_added_turb(x)
        gaussian_ = (
            1
            / (8 * self.sigma**2)
            * np.exp(-(((y - yc) ** 2 + z**2) / (2 * self.sigma**2 * d**2)))
        )

        return gaussian_ * np.nan_to_num(WATI)
    
    def RE_wake_added_turbulence(
        self, x_glob: ArrayLike, y_glob: ArrayLike, z_glob=0
    ) -> ArrayLike:
        """
        Following Niayifar and Porte-Agel 2016 and
        Weisstein, Eric W. "Circle-Circle Intersection."

        Returns wake added turbulence intensity caused by a wake at a particular
        rotor by computing the overlap of a 4 sigma diameter (double wake width)
        top hat with the downstream rotor. """
         
        x, y, z = x_glob - self.x, y_glob - self.y, z_glob - self.z
        R = 2 * self.sigma * self._wake_diameter(x)
        r = 0.5
        d = np.sqrt(y ** 2 + z ** 2)

        WATI = self.centerline_wake_added_turb(x)

        with np.errstate(all = "ignore"):
            wake_overlap = (
                (r ** 2) * np.arccos((d ** 2 + r ** 2 - R ** 2) / (2 * d * r)) + 
                (R ** 2) * np.arccos((d ** 2 + R ** 2 - r ** 2) / (2 * d * R)) -
                0.5 * np.sqrt((-d + r + R)*(d + r - R)*(d - r + R)*(d + r + R))
            )
            wake_overlap[d > r + R] = 0
            wake_overlap[d < R - r] = np.pi * r ** 2
            return (wake_overlap * 4 * np.nan_to_num(WATI)) / np.pi

   
        
    def deficit_and_WATI(
        self, x_glob: ArrayLike, y_glob: ArrayLike, z_glob=0
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        returns both the wake deficit and wake-added turbulence intensity at a point in space.
        """
        x, y, z = x_glob - self.x, y_glob - self.y, z_glob - self.z
        d = self._wake_diameter(x)
        yc = self.centerline(x_glob) - self.y
        du = self._du(x, wake_diameter=d)
        WATI = self.centerline_wake_added_turb(x)
        gaussian_ = (
            1
            / (8 * self.sigma**2)
            * np.exp(-(((y - yc) ** 2 + z**2) / (2 * self.sigma**2 * d**2)))
        )

        return gaussian_ * du, gaussian_ * WATI

    def line_deficit(self, x_glob: np.array, y_glob: np.array):
        """
        Returns the deficit at hub height averaged along a lateral line of
        length 1, centered at (x, y).
        """
        x, y = x_glob - self.x, y_glob - self.y

        d = self._wake_diameter(x)
        yc = self.centerline(x_glob) - self.y
        du = self._du(x, wake_diameter=d)

        erf_plus = erf((y + 0.5 - yc) / (np.sqrt(2) * self.sigma * d))
        erf_minus = erf((y - 0.5 - yc) / (np.sqrt(2) * self.sigma * d))

        deficit_ = np.sqrt(2 * np.pi) * d / (16 * self.sigma) * (erf_plus - erf_minus)

        return deficit_ * du


class GaussianWakeModel(WakeModel):
    def __init__(self, sigma=0.25, kw=0.07, xmax: float = 100.0):
        self.sigma = sigma
        self.kw = kw
        self.xmax = xmax

    def __call__(
        self, x: float, y: float, z: float, rotor_sol: "RotorSolution", TIamb: float = None
    ) -> GaussianWake:
        return GaussianWake(
            x,
            y,
            z,
            rotor_sol,
            sigma=self.sigma,
            kw=self.kw,
            TIamb=TIamb,
            xmax=self.xmax,
        )
    
class VariableKwGaussianWakeModel(WakeModel):
    def __init__(self, a: float = 0.7516, b: float = 0.0196, sigma: float = 0.25):
        self.a = a
        self.b = b
        self.sigma = sigma

    def __call__(self, x, y, z, rotor_sol: "RotorSolution", TIamb: float = None) -> GaussianWake:
        kw = self.a * rotor_sol.TI + self.b
        return GaussianWake(x, y, z, rotor_sol, sigma=self.sigma, kw=kw, TIamb=TIamb)