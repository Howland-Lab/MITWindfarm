from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import cumulative_trapezoid
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
    def wake_added_turbulence(
        self, x: ArrayLike, y: ArrayLike, z: ArrayLike
    ) -> ArrayLike:
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
        sigma: float = 0.25,
        kw: float = 0.07,
        TIamb: float = None,
        xmax: float = 100.0,
        dx: float = 0.05,
        WATI_sigma_multiplier: float = 1.0,
    ):
        self.x, self.y, self.z = x, y, z
        self.rotor_sol = rotor_sol
        self.sigma, self.kw = sigma, kw
        self.WATI_sigma_multiplier = WATI_sigma_multiplier
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
        _yc = cumulative_trapezoid(-dv, dx=dx, initial=0)

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
    
    def niayifar_deficit(self, x_glob: ArrayLike, y_glob: ArrayLike, z_glob=0) -> ArrayLike:
        """
        Solves Eq. C1 where the wake deficit is defined relative to the
        incident rotor wind speed following Niayifar (2016) Energies.
        """
        x, y, z = x_glob - self.x, y_glob - self.y, z_glob - self.z
        d = self._wake_diameter(x)
        yc = self.centerline(x_glob) - self.y
        du = 0.5 * (self.rotor_sol.REWS - self.rotor_sol.u4) / d**2 * (1 + erf(x / (np.sqrt(2) / 2)))
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
        points in space. Laterally smeared with the gaussian twice as wide as
        the wake deficit model. as recommended by Niayifar and Porte-Agel 2016
        """
        x, y, z = x_glob - self.x, y_glob - self.y, z_glob - self.z
        d = self._wake_diameter(x)
        yc = self.centerline(x_glob) - self.y
        WATI = self.centerline_wake_added_turb(x)

        _gaussian = (
            1
            / (8 * (self.WATI_sigma_multiplier * self.sigma) ** 2)
            * np.exp(
                -(
                    ((y - yc) ** 2 + z**2)
                    / (2 * (self.WATI_sigma_multiplier * self.sigma) ** 2 * d**2)
                )
            )
        )

        return _gaussian * np.nan_to_num(WATI)

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
    def __init__(
        self, sigma=0.25, kw=0.07, WATI_sigma_multiplier=1.0, xmax: float = 100.0
    ):
        self.sigma = sigma
        self.kw = kw
        self.xmax = xmax
        self.WATI_sigma_multiplier = WATI_sigma_multiplier

    def __call__(
        self, x, y, z, rotor_sol: "RotorSolution", TIamb: float = None
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
            WATI_sigma_multiplier=self.WATI_sigma_multiplier,
        )


class VariableKwGaussianWakeModel(WakeModel):
    """
    Gaussian wake model which adjust the wake spreading rate (kw) based on the
    Ctprime and the TI experienced by the wake-generating turbine.

    Follows the linear relation:

    kw = a * TI + b * Ctprime + c

    where coefficients a, b, and c are provided at initialization.
    """

    def __init__(
        self,
        a: float,
        b: float,
        c: float,
        sigma: float = 1 / np.sqrt(8),
        WATI_sigma_multiplier=1.0,
        xmax: float = 100.0,
    ):
        self.a = a
        self.b = b
        self.c = c
        self.sigma = sigma
        self.xmax = xmax
        self.WATI_sigma_multiplier = WATI_sigma_multiplier

    def __call__(
        self, x, y, z, rotor_sol: "RotorSolution", TIamb: float = None
    ) -> GaussianWake:
        kw = self.a * rotor_sol.TI + self.b * rotor_sol.Ctprime + self.c
        return GaussianWake(
            x,
            y,
            z,
            rotor_sol,
            sigma=self.sigma,
            kw=kw,
            TIamb=TIamb,
            xmax=self.xmax,
            WATI_sigma_multiplier=self.WATI_sigma_multiplier,
        )
