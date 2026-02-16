"""
Wake model following Bastankhah and Porte-Agel, JFM (2016)
"""


import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, TYPE_CHECKING
from .Wake import Wake, WakeModel

if TYPE_CHECKING:
    from .Rotor import RotorSolution
    from .Windfield import Windfield


class GaussBPWakeModel(WakeModel):
    """
    Defines a Gaussian wake model based on Bastankhah and Porte-Agel (2016).
    __init__: 
        - Args
            - kw: float, linear wake expansion coefficient (default: 0.04)
            - R: float, rotor radius (default: 0.5)
            - windfield: windfield for veer deformation (default: None)
    __call__: function to create a BP2016Wake instance called by the wake model solver
        - Args
            - x: float, x-coordinate of the turbine
            - y: float, y-coordinate of the turbine
            - z: float, z-coordinate of the turbine
            - rotor_sol: RotorSolution, solution object containing rotor parameters
            - TIamb: float, ambient turbulence intensity (default: None)
        - Returns:
            - BP2016Wake instance with the specified parameters.
    """
    def __init__(
        self,
        kw: float = 0.04,
        R: float = 0.5,
        windfield: Optional["Windfield"] = None,
        couple_rotor_x0: bool = False,
        x0: float = None,
        astar: float = 2.32,
        bstar: float = 0.154,
    ):
        """
        Defines a Gaussian wake model based on Bastankhah and Porte-Agel (2016).

        - Args
            - kw: float, linear wake expansion coefficient (default: 0.04)
            - R: float, rotor radius (default: 0.5)
            - windfield: windfield for veer deformation (default: None)
            - couple_rotor_x0: bool, whether to couple near-wake length to rotor (default: False)
            - x0: float, near-wake length. Used if couple_rotor_x0 is False (default: None)
            - astar: float, x0 tuning parameter alpha* (default: 2.32)
            - bstar: float, x0 tuning parameter beta* (default: 0.154)
        """
        self.kw = kw
        self.R = R
        self.windfield = windfield
        self.couple_rotor_x0 = couple_rotor_x0
        self.x0 = x0
        self.astar = astar
        self.bstar = bstar

    def __call__(
        self, x, y, z, rotor_sol: "RotorSolution", TIamb: float = None
    ) -> "BP2016Wake":
        """
        Function to create a BP2016Wake instance called by the wake model solver

        - Args
            - x: float, x-coordinate of the turbine
            - y: float, y-coordinate of the turbine
            - z: float, z-coordinate of the turbine
            - rotor_sol: RotorSolution, solution object containing rotor parameters
            - TIamb: UNUSED (legacy) - use rotor_sol.TI instead
        - Returns:
            - BP2016Wake instance with the specified parameters.
        """
        Ct = rotor_sol.Ct
        x0 = 0.2 * np.sqrt(0.5 * (1 + np.sqrt(1 - Ct)) / np.sqrt(1 - Ct)) / self.kw
        return BP2016Wake(
            x,
            y,
            z,
            rotor_sol,
            ky=self.kw,
            kz=self.kw,
            TI=rotor_sol.TI,  # deprecate TIamb argument
            x0=rotor_sol.extra.x0 if self.couple_rotor_x0 else self.x0,
            theta0=None,
            astar=self.astar,
            bstar=self.bstar,
            d=self.R * 2,
        )


class BP2016Wake(Wake):
    """
    Yawed Gaussian wake model described in Bastankhah and Porte-Agel (2016).
    """

    def __init__(
        self,
        x: float, 
        y: float,
        z: float,
        rotor_sol: "RotorSolution",
        ky: float = 0.035,
        kz: Optional[float] = None,
        TI: float = 0.05,
        x0: float = None,
        theta0: float = None,
        astar: float = 2.32,
        bstar: float = 0.154,
        d: float = 1,
    ):
        """
        Note some definitions: 
            self.ct (float): Rotor thrust coefficient, non-dimensionalized to
                pi/8 d**2 rho u_h^2. This is consistent with Unified Momentum
                model (Liew et al (2024) but not with Shapiro et al. (2018)
            self.yaw (float): Rotor yaw angle (radians), CW positive.

        Args:
            x (float): x-coordinate of the turbine
            y (float): y-coordinate of the turbine
            z (float): z-coordinate of the turbine
            ky (float): Wake spreading parameter. Defaults to 0.07.
            kz (float, optional): Wake spreading parameter. Defaults to None.
            TI (float): Turbulence intensity. Defaults to 0.05.
            x0 (float, optional): Near-wake length. If None, calculated
                with eq. 7.3. Defaults to None.
            theta0 (float, optional): Initial wake deflection angle. If None, calculated
                with eq. 6.12. Defaults to None.
            astar (float, optional): alpha^* tuning parameter. Defaults to 2.32.
            bstar (float, optional): beta^* tuning parameter. Defaults to 0.154.
            d (float, optional): non-dimensionalizing value for diameter. Defaults to 1.
        """
        self.x, self.y, self.z = x, y, z
        self.rotor_sol = rotor_sol
        self.ct = rotor_sol.Ct / rotor_sol.REWS**2
        self.yaw = -rotor_sol.yaw  # BP2016 uses CW positive sign convention for yaw
        self.ky = ky
        if kz is None:
            self.kz = ky
        else:
            self.kz = kz
        self.TIamb = TI
        self.astar = astar
        self.bstar = bstar
        self.d = d
        self.x0 = self.calc_x0() if x0 is None else x0
        self.theta0 = self.calc_theta0() if theta0 is None else theta0

    def calc_theta0(self):
        """
        Solves eq. 6.12
        """
        theta_0 = (
            0.3
            * self.yaw
            / np.cos(self.yaw)
            * (1 - np.sqrt(1 - self.ct * np.cos(self.yaw)))
        )
        return theta_0

    def calc_x0(self):
        """
        Solves eq. 7.3
        """
        x0 = self.d * np.cos(self.yaw) * (1 + np.sqrt(1 - self.ct)) / \
            (np.sqrt(2) * (self.astar * self.TIamb + self.bstar * (1 - np.sqrt(1 - self.ct))))
        print(x0, self.ct, self.astar, self.TIamb, self.bstar)
        return x0

    def sigma_y(
        self,
        x: np.array,
    ):
        """
        Solves eq. 7.2a, non-dimensionalized by d
        """
        x = np.atleast_1d(x)
        sigma_y0 = np.cos(self.yaw) / np.sqrt(8)
        sigma_y = self.ky * (x - self.x0) + sigma_y0
        sigma_y[x < self.x0] = sigma_y0
        return sigma_y

    def sigma_z(
        self,
        x: np.array,
    ):
        """
        Solves eq. 7.2b, non-dimensionalized by d
        """
        x = np.atleast_1d(x)
        sigma_z0 = 1 / np.sqrt(8)
        sigma_z = self.kz * (x - self.x0) + sigma_z0
        sigma_z[x < self.x0] = sigma_z0
        return sigma_z

    def centerline(
        self,
        x: np.array,
    ):
        """
        Solves eq. 7.4
        """
        x = np.atleast_1d(x)
        d = self.d
        t0 = self.theta0
        ct = self.ct
        cos = np.cos(self.yaw)
        A1 = 1.6 * np.sqrt(
            8 * self.sigma_y(x) * self.sigma_z(x) / d**2 / cos
        )  # tmp variable

        delta = t0 * self.x0 + d * t0 / 14.7 * np.sqrt(cos / self.ky / self.kz / ct) * (
            2.9 + 1.3 * np.sqrt(1 - ct) - ct
        ) * np.log(
            (1.6 + np.sqrt(ct)) * (A1 - np.sqrt(ct))
            / ((1.6 - np.sqrt(ct)) * (A1 + np.sqrt(ct)))
        )

        delta = np.atleast_1d(delta)
        delta[x < self.x0] = t0 * x[x < self.x0]
        delta[x < 0] = 0
        return delta
    
    def _gaussian(self, x_glob, y_glob, z_glob, sigma_multiplier=1):
        """
        Defines a Gaussian profile.
        """
        x = x_glob - self.x  # local coordinates
        y = y_glob - self.y
        z = z_glob - self.z
        sigma_y = self.sigma_y(x) * sigma_multiplier
        sigma_z = self.sigma_z(x) * sigma_multiplier
        delta = self.centerline(x)
        
        return (
            np.exp(-0.5 * ((y - delta) / sigma_y) ** 2)
            * np.exp(-0.5 * (z / sigma_z) ** 2)
        )

    def deficit(self, x_glob: ArrayLike, y_glob: ArrayLike, z_glob=0) -> ArrayLike:
        """
        Computes wake deficit (eq. 7.1)
        """
        x = x_glob - self.x  # local coordinates
        sigma_y = self.sigma_y(x)
        sigma_z = self.sigma_z(x)
        
        radical = 1 - self.ct * np.cos(self.yaw) / (8 * sigma_y * sigma_z)
        Cx = 1 - np.sqrt(np.clip(radical, 0, None))  # avoid negative values inside sqrt
        Cx[x < 0] = 0  # no upstream wakes!
        delta_u = Cx * self._gaussian(x_glob, y_glob, z_glob)
        
        return delta_u

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

    def wake_added_turbulence(
            self, x_glob: ArrayLike, y_glob: ArrayLike, z_glob: 0
    ) -> ArrayLike: 
        """
        Computes wake-added turbulence with the Crespo-Hernandez model and 
        laterally smeared twice the Gaussian width as recommended by
        Niayifar and Porté-Agel (2016).
        """
        gaussian = self._gaussian(
            x_glob, y_glob, z_glob, sigma_multiplier=2.0
        )
        return self.centerline_wake_added_turb(x_glob - self.x) * gaussian
