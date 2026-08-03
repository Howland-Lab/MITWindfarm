"""
Vortex wake model from Bastankhah et al. JFM (2022) and extended to
wind veer effects by Narasimhan, Gayme, and Meneveau JRSE (2025).

Note that the Coupled Ekman-surface layer ABL model is not included
here for flexibility of the model solution to take arbitrary
wind profiles. A separate Windfield object, which takes the inputs
to the Coupled Ekman-surface layer ABL model can be used to fully
describe the wake model in the paper.
"""

import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, TYPE_CHECKING
from .Wake import Wake, WakeModel

if TYPE_CHECKING:
    from .Rotor import RotorSolution
    from .Windfield import Windfield


class VortexWakeModel(WakeModel):
    """
    Defines a vortex wake model based on the work of Narasimhan, Gayme, and Meneveau (2025)
    which extends the work of Bastankhah et al. JFM (2022) to include wind veer effects.

    __init__: 
        - Args
            - kw: float, linear wake expansion coefficient (default: 0.04)
            - R: float, rotor radius (default: 0.5)
            - alpha: float, shape parameter for wake deformation (default: 1.263)
            - ustar: float, friction velocity for ground effect modeling (default: None)
            - windfield: windfield for veer deformation (default: None)
            - z_wall: bool, whether to include ground effect modeling (default: None - no wall)
    __call__: function to create a VortexWake instance called by the wake model solver
        - Args
            - x: float, x-coordinate of the turbine
            - y: float, y-coordinate of the turbine
            - z: float, z-coordinate of the turbine
            - rotor_sol: RotorSolution, solution object containing rotor parameters
            - TIamb: float, ambient turbulence intensity (default: None)
        - Returns:
            - VortexWake instance with the specified parameters.
    """
    def __init__(
        self,
        kw: float = 0.04,
        R: float = 0.5,
        alpha: float = 1.263,
        ustar: Optional[float] = None,
        windfield: Optional["Windfield"] = None,
        z_wall: bool = None,
    ):
        """
        Defines a vortex wake model based on the work of Narasimhan, Gayme, and Meneveau (2025)
        which extends the work of Bastankhah et al. JFM (2022) to include wind veer effects.

        - Args
            - kw: float, linear wake expansion coefficient (default: 0.04)
            - R: float, rotor radius (default: 0.5)
            - alpha: float, shape parameter for wake deformation (default: 1.263)
            - ustar: float, friction velocity for ground effect modeling (default: None)
            - windfield: windfield for veer deformation (default: None)
            - z_wall: bool, whether to include ground effect modeling (default: None - no wall)
        """
        self.kw = kw
        self.R = R
        self.alpha = alpha
        self.ustar = ustar
        self.windfield = windfield
        self.z_wall = z_wall

    def __call__(
        self, x, y, z, rotor_sol: "RotorSolution", TIamb: float = None
    ) -> "VortexWake":
        """
        Function to create a VortexWake instance called by the wake model solver

        - Args
            - x: float, x-coordinate of the turbine
            - y: float, y-coordinate of the turbine
            - z: float, z-coordinate of the turbine
            - rotor_sol: RotorSolution, solution object containing rotor parameters
            - TIamb: UNUSED (legacy) - use rotor_sol.TI instead
        - Returns:
            - VortexWake instance with the specified parameters.
        """
        return VortexWake(
            x,
            y,
            z,
            rotor_sol,
            kw=self.kw,
            alpha=self.alpha,
            R=self.R,
            ustar=self.ustar,
            windfield=self.windfield,
            z_wall=self.z_wall,
            TIamb=rotor_sol.TI,  # this is rotor TI
        )


class VariableVortexWakeModel(VortexWakeModel):
    """
    Defines a vortex wake model based on the work of Narasimhan, Gayme, and Meneveau (2025)
    which extends the work of Bastankhah et al. JFM (2022) to include wind veer effects.

    Here, we parameterize the variable wake spreading rate through the relationship
    proposed in Niayifar and Porte-Agel (2016): 
        kw = 0.3837 * TIamb + 0.003678

    __init__: 
        - Args
            - a: float, TI dependence on kw (default: 0.3837)
            - b: float, constant offset on kw (default: 0.003678)
            - R: float, rotor radius (default: 0.5)
            - alpha: float, shape parameter for wake deformation (default: 1.263)
            - ustar: float, friction velocity for ground effect modeling (default: None)
            - windfield: windfield for veer deformation (default: None)
            - z_wall: bool, whether to include ground effect modeling (default: None - no wall)
    __call__: function to create a VortexWake instance called by the wake model solver
        - Args
            - x: float, x-coordinate of the turbine
            - y: float, y-coordinate of the turbine
            - z: float, z-coordinate of the turbine
            - rotor_sol: RotorSolution, solution object containing rotor parameters
            - TIamb: float, ambient turbulence intensity (default: None)
        - Returns:
            - VortexWake instance with the specified parameters.
    """
    def __init__(
        self,
        a: float = 0.3837,
        b: float = 0.003678,
        R: float = 0.5,
        alpha: float = 1.263,
        ustar: Optional[float] = None,
        windfield: Optional["Windfield"] = None,
        z_wall: bool = None,
    ):
        """
        Defines a vortex wake model based on the work of Narasimhan, Gayme, and Meneveau (2025)
        which extends the work of Bastankhah et al. JFM (2022) to include wind veer effects.

        - Args
            - a: float, TI dependence on kw (default: 0.3837)
            - b: float, constant offset on kw (default: 0.003678)
            - R: float, rotor radius (default: 0.5)
            - alpha: float, shape parameter for wake deformation (default: 1.263)
            - ustar: float, friction velocity for ground effect modeling (default: None)
            - windfield: windfield for veer deformation (default: None)
            - z_wall: bool, whether to include ground effect modeling (default: None - no wall)
        """
        super().__init__(
            kw=None,
            R=R,
            alpha=alpha,
            ustar=ustar,
            windfield=windfield,
            z_wall=z_wall,
        )
        self.a = a
        self.b = b

    def __call__(
        self, x, y, z, rotor_sol: "RotorSolution", TIamb: float = None
    ) -> "VortexWake":
        return VortexWake(
            x, y, z,
            rotor_sol,
            kw=self.a * rotor_sol.TI + self.b,  # compute on-the-fly
            alpha=self.alpha,
            R=self.R,
            ustar=self.ustar,
            windfield=self.windfield,
            z_wall=self.z_wall,
            TIamb=rotor_sol.TI,
        )


class VortexWake(Wake):
    """
    Vortex wake object based on the work of Narasimhan, Gayme, and Meneveau (2025).
    """

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        rotor_sol: "RotorSolution",
        kw: float = 0.04,
        alpha: float = 1.263,
        R: float = 0.5,
        ustar: Optional[float] = None,
        windfield: Optional["Windfield"] = None,
        z_wall: bool = True,
        TIamb: float = None,
    ):
        self.x, self.y, self.z = x, y, z
        self.rotor_sol = rotor_sol
        self.Ct = self.rotor_sol.Ct / self.rotor_sol.REWS**2
        self.kw = kw
        self.alpha = alpha
        self.Astar = self.A_star()  # A_* parameter for the vortex wake model
        self.R = R
        self.ustar = ustar
        self.windfield = windfield
        self.z_wall = z_wall
        self.TIamb = TIamb

    def A_star(self):
        """
        Returns the A_* parameter for the vortex wake model using
        Eq. (4) from Narasimhan, Gayme, and Meneveau (2025).

        Note that the RotorSolution C_T thrust coefficent does not
        include cos(yaw)^2 in the denominator, so the equation for
        A_star omits the cos(yaw)^2 terms multiplied by C_T.
        """
        # return (1 + np.sqrt(1 - self.Ct)) / 2 / np.sqrt(1 - self.Ct)
        an = self.rotor_sol.an / self.rotor_sol.REWS
        u4 = self.rotor_sol.u4 / self.rotor_sol.REWS
        return (1 - an) / u4

    def du(self, x):
        """
        Returns the maximum wake velocity deficit at a downstream position
        `x` which is with respect to the turbine x-location; Eq. 2.
        """
        x = np.atleast_1d(x)
        radical = 1 - self.Ct * np.cos(self.rotor_sol.yaw) / (
            2 * self.sigmatildesq(x) / self.R**2
        )
        Cx = 1 - np.sqrt(np.clip(radical, 0, None))
        # x0 = self.x0()
        # Cx[x <= x0] = 2 * self.rotor_sol.an / self.rotor_sol.REWS  # This is in the paper

        # use this instead:  (generalizes to Unified model)
        # Cx = np.clip(Cx, 0, 1 - self.rotor_sol.u4 / self.rotor_sol.REWS)
        Cx[x < 0] = 0  # no wake deficit in front of the turbine
        return Cx

    def sigmatildesq(self, x):
        """
        Returns the sigma_tilde^2 parameter (wake width) as a function of
        downstream distance `x`, which is with respect to the turbine
        x-location; Eq. 3.
        """
        res = (self.kw * x + 0.4 * self.R * np.sqrt(self.Astar)) * (
            self.kw * x
            + 0.4 * self.R * np.sqrt(self.Astar) * np.cos(self.rotor_sol.yaw)
        )
        return res

    def x0(self):
        """
        Computes near-wake length x0 (dimensional). Note that x0 in the
        rotor solution is unused. Solves Eq. 6
        """
        cos = np.cos(self.rotor_sol.yaw)
        u4 = self.rotor_sol.u4 / self.rotor_sol.REWS
        # u4 = 2 * self.rotor_sol.an / self.rotor_sol.REWS  # This is in the paper
        x0_dim = (
            1
            / (5 * self.kw)
            * (
                np.sqrt(
                    self.R**2 * self.Astar * (1 - cos) ** 2
                    + 25 * self.R**2 / 2 * self.Ct * cos / (1 - (1 - u4) ** 2)
                )
                + self.R * np.sqrt(self.Astar) * (1 - cos)
            )
        ) - 2 / (5 * self.kw) * self.R * np.sqrt(self.Astar)
        return x0_dim

    def sigma(self, x, z, theta):
        """
        Computes the gaussian wake width as a function of `x`
        and `z` (distances w.r.t. the turbine),
        and the local polar angle `theta`. Solves Eq. 7
        """
        return self.kw * x + 0.4 * self.xi(x, z, theta)

    def xi(self, x, z, theta):
        """
        Computes the shape function xi as function of downstream
        distance `x` (downstream distance w.r.t. the turbine), the local
        vertical position `z`, and the local polar angle `theta`.
        Solves Eq. 9
        """
        xi_0 = self.xi_0(theta)
        xi_hat = self.xi_hat(x, z, theta)
        return xi_0 * xi_hat

    def xi_0(self, theta):
        """
        Returns the xi_0 parameter (initial elliptical wake shape for a
        yawed turbine) from Eq. 10.
        """
        return (
            self.R
            * np.sqrt(self.Astar)
            * np.abs(np.cos(self.rotor_sol.yaw))
            / np.sqrt(1 - np.sin(self.rotor_sol.yaw) ** 2 * np.sin(theta) ** 2)
        )

    def xi_hat(self, x, z, theta):
        """
        Returns the time-varying (spatially evolving) shape function
        from Eq. 11.
        """
        t_hat = self.t_hat(x, z)
        xi_hat = 1 - self.alpha * (
            0.5 * np.tanh(t_hat**2 / (4 * self.alpha)) * np.cos(2 * theta)
            - 1 / 4 * np.tanh(t_hat**3 / (8 * self.alpha)) * np.cos(3 * theta)
            - 5 / 48 * np.tanh(t_hat**4 / (16 * self.alpha)) * np.cos(2 * theta)
            + 7 / 48 * np.tanh(t_hat**4 / (16 * self.alpha)) * np.cos(4 * theta)
        )

        return xi_hat

    def t_hat(self, x, z):
        """
        Returns dimensionless auxilliary time variable t_hat from Eq. 12
        """
        if self.rotor_sol.yaw == 0:
            return 0

        if self.ustar is not None and self.ustar > 0:
            t_hat = (
                -1.44
                * self.rotor_sol.REWS
                / (self.ustar * np.sqrt(self.Astar))
                * self.Ct
                * np.sin(self.rotor_sol.yaw)
            ) * (1 - np.exp(-0.35 * self.ustar / self.Uz(z) * x / self.R))
        else:
            gamma_b0 = -0.5 * self.rotor_sol.REWS * self.Ct * np.sin(self.rotor_sol.yaw)
            t = x / self.Uz(z)  # "real" advection time
            t_hat = gamma_b0 * t / (self.R * np.sqrt(self.Astar))

        return t_hat

    def Uz(self, z, clip=1e-6):
        """
        Returns the wind speed at height `z` from the windfield,
        which is defined with respect to the turbine z-location. If no
        windfield is provided, returns the rotor-equivalent wind speed.
        """
        if self.windfield is not None:
            z_glob = z + self.z  # transform to global coordinates
            return np.clip(self.windfield.wsp(0, 0, z_glob), clip, None)
        else:
            return self.rotor_sol.REWS

    def wdir(self, z):
        """
        Returns wind direction at height `z` from the windfield, which
        is defined with respect to the turbine z-location. If no
        windfield is provided, returns 0 (no veer).
        """
        if self.windfield is not None:
            # transform back to global coordinates
            return np.clip(
                self.windfield.wdir(0, 0, z + self.z), np.pi * -0.45, np.pi * 0.45
            )
        else:
            return 0

    def centerline(self, x, z):
        """
        Returns the wake centerline position at a downstream location
        `x` due to yawed wake deflection and wind veer. Solves Eq. 17

        Both `x` and `z` are with respect to the turbine x-location.
        """
        yc_veer = x * np.tan(self.wdir(z))
        return self.yc_hat(x, z) * self.R * np.sqrt(self.Astar) + yc_veer

    def yc_hat(self, x, z):
        """
        Returns the time-varying (spatially evolving) centerline position
        from Eq. 18. Inputs `x` and `z` are with respect to the turbine.

        The second term is only included if the ground effect is modeled.
        Note that in the ground effect term, self.z is multiplied by two
        because `z` is in local (turbine) coordinates, so self.z is added once
        to transform to global coordinates, and then again to account for the
        zhub offset (distance from ground)
        """
        t_hat = self.t_hat(x, z)
        yc_hat = (
            (
                (np.pi - 1) * np.abs(t_hat) ** 3
                + 2 * np.sqrt(3) * np.pi**2 * t_hat**2
                + 48 * (np.pi - 1)**2 * np.abs(t_hat)
            )
            / (
                2 * np.pi * (np.pi - 1) * t_hat**2
                + 4 * np.sqrt(3) * np.pi**2 * np.abs(t_hat)
                + 96 * (np.pi - 1) ** 2
            )
            * np.sign(t_hat)
        )
        if self.z_wall is not None:
            # note: denominator has z + (self.z - self.z_wall) * 2 because `z` is in
            # local (turbine) coordinates, so self.z is added once to transform to global
            # coordinates, and then again to account for the zhub offset (distance from ground)
            yc_hat -= (
                2
                * t_hat
                / np.pi
                / (((z + (self.z - self.z_wall) * 2) / (self.R * np.sqrt(self.Astar))) ** 2 - 1)
            )

        return yc_hat

    def deficit(self, x_glob, y_glob, z_glob):
        """
        Computes the wake velocity deficit at the global x, y, z coordinates.
        Returns Eq. 1
        """
        x = x_glob - self.x
        z = z_glob - self.z
        Cx = self.du(x)
        yc = self.centerline(x, z)
        y = y_glob - yc - self.y
        theta = np.arctan2(z, y)
        sigma = self.sigma(x, z, theta)
        gaussian = np.exp(-0.5 * ((y**2 + z**2) / sigma**2))
        return Cx * gaussian

    def niayifar_deficit(self, *args):
        return self.deficit(*args) * self.rotor_sol.REWS

    def wake_added_turbulence(
        self, x_glob: ArrayLike, y_glob: ArrayLike, z_glob=0
    ) -> ArrayLike:
        """
        Returns wake added turbulence intensity caused by a wake at particular
        points in space. Laterally smeared with the gaussian twice as wide as
        the wake deficit model as recommended by Niayifar and Porte-Agel (2016).
        """
        x = x_glob - self.x
        z = z_glob - self.z
        yc = self.centerline(x, z)
        y = y_glob - yc - self.y
        theta = np.arctan2(z, y)
        sigma = self.sigma(x, z, theta)
        gaussian = np.exp(-0.5 * ((y**2 + z**2) / (sigma * 2)**2))

        WATI = self.centerline_wake_added_turb(x)
        return gaussian * np.nan_to_num(WATI)
    
    def centerline_wake_added_turb(self, x: ArrayLike) -> ArrayLike:
        """
        Returns the centerline wake-added turbulence intensity (WATI) based on
        the model by Crespo and Hernandez (1996). Input `x` is the downstream
        distance with respect to the turbine x-location.
        """
        x = np.atleast_1d(x)
        if self.windfield is not None and self.TIamb is None: 
            # NOTE: this is not the same as self.rotor_sol.RETI, which includes upstream wakes
            # But the dependence on TIamb is weak (only ~10% difference between 1% and 20% TI)
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
