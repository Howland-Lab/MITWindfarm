"""
Defines the skewed Gaussian model for the Abkar et al. (2018) wake model.
"""

import numpy as np
from numpy.typing import ArrayLike
from typing import Union, Optional, TYPE_CHECKING
from .Wake import Wake, WakeModel
from .Windfield import Windfield

if TYPE_CHECKING:
    from .Rotor import RotorSolution


class SkewGaussianWakeModel(WakeModel):
    """
    Defines a skewed Gaussian wake model for the Abkar et al. (2018) wake model.

    Parameters: 
        ky (float): lateral wake spreading coefficient
        kz (float, optional): vertical wake spreading coefficient. Defaults to `ky`.
        WATI_sigma_multiplier (float): multiplier for the wake-added turbulence intensity (WATI)
        xmax (float): maximum downstream distance for the wake -- UNUSED
        alpha_in (float or ArrayLike): veer angle in radians,or a 1D array of veer 
            angles at different heights in the global coordinate system.
        alpha_z (ArrayLike, optional): if `alpha_in` is an array, this is the heights in the local
            coordinate system at which the veer angles are defined.
        base_windfield (Windfield, optional): if provided, overrides `alpha_in` and `alpha_z`.

    Methods: 
        __call__(x, y, z, rotor_sol, TIamb=None): Creates a Wake object at the given turbine coordinates.
    """

    def __init__(
        self,
        ky=0.024,
        kz=None,
        WATI_sigma_multiplier=1.0,
        xmax: float = 100.0,
        alpha_in: Union[float, ArrayLike] = 0,
        alpha_z: Optional[ArrayLike] = None,
        base_windfield: Optional["Windfield"] = None, 
    ):
        self.ky = ky
        self.kz = kz if kz is not None else ky
        self.xmax = xmax
        self.WATI_sigma_multiplier = WATI_sigma_multiplier

        # veer parameters:
        self.alpha_in = alpha_in
        self.alpha_z = None
        if base_windfield is not None:
            self.windfield = base_windfield
        else: 
            self.windfield = None
            if isinstance(self.alpha_in, np.ndarray):
                if alpha_z is None:
                    raise ValueError("alpha_z must be provided if alpha_in is an array.")
                self.alpha_z = alpha_z

    def __call__(
        self, x, y, z, rotor_sol: "RotorSolution", TIamb: float = None
    ) -> "SkewGaussianWake":

        return SkewGaussianWake(
            x,
            y,
            z,
            rotor_sol,
            ky=self.ky,
            kz=self.kz,
            alpha_in=self.alpha_in,
            alpha_z=self.alpha_z,
            windfield=self.windfield, 
            TIamb=TIamb, 
        )


class SkewGaussianWake(Wake):
    """
    Defines the skewed Gaussian wake for the Abkar et al. (2018) equations.
    """

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        rotor_sol: "RotorSolution",
        ky: float = 0.024,
        kz: Union[float, None] = None,
        alpha_in: Union[float, ArrayLike, str] = 0,
        alpha_z: Optional[ArrayLike] = None,
        WATI_sigma_multiplier: float = 1.0,
        windfield: Optional["Windfield"] = None,
        TIamb: Optional[float] = None,
    ):
        self.x, self.y, self.z = x, y, z
        self.rotor_sol = rotor_sol
        self.ky = ky
        self.kz = kz if kz is not None else ky
        self.WATI_sigma_multiplier = WATI_sigma_multiplier
        self.TIamb = TIamb

        # veer parameters: TODO - FORMALIZE THIS 
        self.windfield = windfield
        self.alpha_in = alpha_in
        if alpha_in == "rotor":
            self.windfield = rotor_sol.extra[
                "windfield"
            ]  # need to compute this in `self.deficit`

        else:
            if isinstance(self.alpha_in, np.ndarray):
                self.alpha_z = alpha_z
                if self.alpha_z is None:
                    raise ValueError(
                        "alpha_z must be provided if alpha_in is an array."
                    )
                else:
                    self.alpha_z = np.asarray(self.alpha_z)

    def wdir(self, z: ArrayLike) -> ArrayLike:
        """
        Returns wind direction at height `z` from the windfield, which
        is defined with respect to the turbine z-location. If no
        windfield is provided, returns 0 (no veer).

        Parameters
            z (ArrayLike): height in the local coordinate system
        """
        if self.windfield is not None:
            # transform back to global coordinates
            return np.clip(
                self.windfield.wdir(0, 0, z + self.z), np.pi * -0.45, np.pi * 0.45
            )
        else:
            if isinstance(self.alpha_in, np.ndarray):
                return np.interp(z, self.alpha_z, self.alpha_in)
            else:
                return self.alpha_in * z

    def deficit(
        self, x_glob: ArrayLike, y_glob: ArrayLike, z_glob: ArrayLike
    ) -> ArrayLike:
        """Computes velocity deficit field"""
        x, y, z = x_glob - self.x, y_glob - self.y, z_glob - self.z

        Ct = self.rotor_sol.Ct / self.rotor_sol.REWS**2
        eps = 0.2 * np.sqrt(0.5 * (1 + np.sqrt(1 - Ct)) / np.sqrt(1 - Ct))
        sigma_y = self.ky * x + eps
        sigma_z = self.kz * x + eps

        alpha_in = self.wdir(z)  # veer angle at height z

        u4 = self.rotor_sol.u4 / self.rotor_sol.REWS
        radical = np.clip(1 - Ct / (8 * sigma_y * sigma_z), 0, None)  # clip to avoid NaN
        du = 1 - np.sqrt(radical)
        du = np.clip(du, 0, None) * (x >= 0)  # clip for near wake fix and to prevent wakes upstream
        gaussian = np.exp(
            -0.5 * (((y - x * np.tan(alpha_in)) / sigma_y) ** 2 + (z / sigma_z) ** 2)
        )
        return gaussian * du

    def niayifar_deficit(self, *args):
        """
        Computes velocity deficit field using the Niayifar et al. (2016) model.
        """
        return self.deficit(*args) * self.rotor_sol.REWS

    def centerline(self, x: ArrayLike) -> ArrayLike:
        return 0  # no deflection model currently

    def centerline_wake_added_turb(self, x: ArrayLike) -> ArrayLike:
        """
        Returns the centerline wake-added turbulence intensity (WATI) based on
        the model by Crespo and Hernandez (1996). Input `x` is the downstream
        distance with respect to the turbine x-location.
        """
        x = np.atleast_1d(x)
        if self.windfield is not None and self.TIamb is None:
            # NOTE: this is not the same as self.rotor_sol.RETI, which includes upstream wakes
            TIamb = self.windfield.TI(self.x, self.y, self.z)
            self.TIamb = TIamb

        if self.TIamb is None or self.TIamb == 0.0:
            return np.zeros_like(x)

        with np.errstate(all="ignore"):
            WATI = (
                0.73
                * (self.rotor_sol.an / self.rotor_sol.REWS) ** 0.8325
                * self.TIamb ** (-0.0325)
                * np.maximum(x, 0.1) ** (-0.32)
            )
        WATI[x < 0.1] = 0.0
        return WATI

    def wake_added_turbulence(
        self, x_glob: ArrayLike, y_glob: ArrayLike, z_glob: ArrayLike
    ) -> ArrayLike:
        """
        Returns wake added turbulence intensity caused by a wake at particular
        points in space. Laterally smeared with the gaussian twice as wide as
        the wake deficit model. as recommended by Niayifar and Porte-Agel 2016
        """
        x, y, z = x_glob - self.x, y_glob - self.y, z_glob - self.z

        Ct = self.rotor_sol.Ct
        eps = 0.2 * np.sqrt(0.5 * (1 + np.sqrt(1 - Ct)) / np.sqrt(1 - Ct))
        sigma_y = self.ky * x + eps
        sigma_z = self.kz * x + eps

        if self.windfield is not None:
            alpha_in = self.windfield.wdir(x, y, z)
        else:
            if isinstance(self.alpha_in, np.ndarray):
                alpha_in = np.interp(z, self.alpha_z, self.alpha_in)
            else:
                alpha_in = self.alpha_in * z

        gaussian = np.exp(
            -0.5 * (((y - x * np.tan(alpha_in)) / (2 * sigma_y)) ** 2 + (z / (2 * sigma_z)) ** 2)
        )
        return gaussian * self.centerline_wake_added_turb(x)
