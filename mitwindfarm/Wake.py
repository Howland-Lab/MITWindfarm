from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING
import warnings

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


class JensenWake(Wake):
    """
    Attributes:
        x: x-position of rotor in global coordinate frame
        y: y-position of rotor in global coordinate frame
        z: z-position of rotor in global coordinate frame
        rotor_sol: Rotor solution
        sigma: Proportionality constant for wake diameter used in Gaussian and thus expected
        by WakeModel class, even though Jensen doesn't require a sigma 
        kw: Constant coefficient for wake growth
        xmax: Maximum x value evaluated
        dx: Interval of x values evaluated
        TIamb: Ambient turbulence intensity

    """
    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        rotor_sol: "RotorSolution",
        sigma: float = None,
        kw: float = 0.07,
        TIamb: float = None,
        xmax: float = 100.0,
        dx: float = 0.05
    ):
        self.x, self.y, self.z = x, y, z
        self.rotor_sol = rotor_sol
        self.sigma, self.kw = sigma, kw
        self.TIamb = TIamb or 0.0
        self.xmax = xmax
        self.dx = dx

        # Precompute centerline far downstream
        self.x_centerline, self.y_centerline, self.z_centerline = self._centerline(xmax, dx)

    def deficit(self, x_glob: ArrayLike, y_glob: ArrayLike, z_glob = 0) -> ArrayLike:
        
        # Into rotor coordinate frame
        x = np.float64(x_glob - self.x)
        y = np.float64(y_glob - self.y)
        z = np.float64(z_glob - self.z)
        yc = self.centerline(x_glob)[0] - self.y
        zc = self.centerline(x_glob)[1] - self.z
        r = np.sqrt((y - yc) ** 2 + (z - zc) ** 2)

        # Calculate wake diameter for each streamwise coordinate
        d = self._wake_diameter(x)

        # Calculate du for each streamwise coordinate
        du = self._du(x, wake_diameter = d)

        # Set deficit to be only inside wake region
        # Handle x is a constant
        if np.shape(x) == ():
            deficit = np.zeros(np.shape(y))

            for ii, _ in np.ndenumerate(deficit):
                if (np.sqrt(r[ii] ** 2) <= 1/2*d) and (x >= 0):
                    deficit[ii] = du
                else:
                    # Outside of wake, there is no deficit (velocity is the same as u_inf)
                    deficit[ii] = 0

        # Handle x is an array
        else:
            deficit = np.zeros(np.shape(x))

            for ii, _ in np.ndenumerate(deficit):
                if (np.sqrt(r[ii] ** 2) <= 1/2*d[ii]) and (x[ii] >= 0):
                    deficit[ii] = du[ii]
                else:
                    # Outside of wake, there is no deficit (velocity is the same as u_inf)
                    deficit[ii] = 0

        return deficit


    def _wake_diameter(self, x: ArrayLike) -> ArrayLike:
        """
        Solves the normalized far-wake diameter
        """

        return 1 + 2 * self.kw * x
    
    def _du(self, x: ArrayLike, wake_diameter: Optional[float] = None) -> ArrayLike:
        """
        Solves for nondimensionalized wake deficit velocity 
        """
        
        d = self._wake_diameter(x) if wake_diameter is None else wake_diameter

        du = (1 - self.rotor_sol.u4 / self.rotor_sol.REWS) / d**2

        return du

    def wake_added_turbulence(self, x_glob, y_glob, z_glob):

        # Placeholder of zeroes
        return np.zeros(np.shape(x_glob))
    
    def centerline(self, x_glob: ArrayLike) -> ArrayLike:
        """
        Interpolates Eq. 6 from Shapiro, Gayme, and Meneveau, 2018 (same as for 
        GaussianWake) for centerline y position in global coordinates        
        """
        x = x_glob - self.x

        # Interpolate previously computed centerlines
        yc_temp = np.interp(x, self.x_centerline, self.y_centerline, left=0)
        zc_temp = np.interp(x, self.x_centerline, self.z_centerline, left=0)

        # Translate centerlines into global coordinates
        yc = yc_temp + self.y
        zc = zc_temp + self.z

        return yc, zc
    
    def _centerline(self, xmax: float, dx: float = 0.05) -> ArrayLike:
        """
        Returns centerline y and z position in rotor coordinates. 

        Based on principle from Shapiro, Gayme, and Meneveau, 2018, the 
        transverse velocity wake recovery should parallel the axial velocity 
        wake recovery. The centerline y position in global coordinates is 
        then computed numerically using the transverse velocity deficit,
        based on Equation 3.9 in Shapiro, Gayme, and Meneveau, 2018.
        """

        _x = np.arange(0, max(xmax, 2 * dx), dx)
        d = self._wake_diameter(_x)

        # Compute centerline y position when d != 0
        d_mask = d > 0
        dv = np.zeros(np.shape(d))
        dv[d_mask] = (- self.rotor_sol.v4 / self.rotor_sol.REWS) / d[d_mask]**2
        _yc = cumulative_trapezoid(-dv, dx=dx, initial=0)

        # Compute centerline z position when d != 0
        dw = np.zeros(np.shape(d))
        dw[d_mask] = (- self.rotor_sol.w4 / self.rotor_sol.REWS) / d[d_mask]**2
        _zc = cumulative_trapezoid(-dw, dx=dx, initial=0)

        return _x, _yc, _zc

    def niayifar_deficit(self, x_glob: ArrayLike, y_glob: ArrayLike, z_glob = 0) -> ArrayLike:
        
        # Calculate deficit
        deficit = self.deficit(x_glob, y_glob, z_glob)

        # Multiply by REWS to get Niayifar deficit
        return self.rotor_sol.REWS * deficit


class JensenWakeModel(WakeModel):
    def __init__(
        self,
        sigma: float = None,
        kw = 0.07,
        xmax: float = 100.0,
        dx: float = 0.05
    ):
        self.sigma = sigma
        self.kw = kw
        self.xmax = xmax
        self.dx = dx

    def __call__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        rotor_sol: "RotorSolution",
        TIamb: float = None
    ) -> JensenWake:
        return JensenWake(
            x,
            y,
            z,
            rotor_sol,
            sigma = self.sigma,
            kw = self.kw,
            TIamb = TIamb,
            xmax = self.xmax,
            dx = self.dx
        )
        

class TurbOParkWake(Wake):
    """
    Attributes:
        x: x-position of rotor in global coordinate frame
        y: y-position of rotor in global coordinate frame
        z: z-position of rotor in global coordinate frame
        rotor_sol: Rotor solution
        TIamb: Ambient turbulence intensity
        A: Wake expansion calibration parameter 
        c_1, c_2: Empirically estimated constants in wake-added turbulence intensity
        xmax: Maximum x value evaluated
        dx: Interval of x values evaluated
    """

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        rotor_sol: "RotorSolution",
        TIamb: float = 0.1,
        A: float = 0.04,
        c_1: float = 1.5,
        c_2: float = 0.8,
        xmax: float = 100.0,
        dx: float = 0.05
    ):
        self.x, self.y, self.z = x, y, z
        self.rotor_sol = rotor_sol
        self.TIamb = TIamb
        self.A = A
        self.c_1 = c_1
        self.c_2 = c_2

    def deficit(self, x_glob: ArrayLike, y_glob: ArrayLike, z_glob = 0):
        """
        Solves Eq. 6 in Pedersen et al. 2022 for wake deficit profile
        """

        x = np.float64(x_glob - self.x)
        y = np.float64(y_glob - self.y)
        z = np.float64(z_glob - self.z)
        yc = self.centerline(x_glob)[0] - self.y
        zc = self.centerline(x_glob)[1] - self.z
        r = np.sqrt((y - yc) ** 2 + (z - zc) ** 2) 

        sigma = self._char_wake_diameter(x)

        # Only compute behind rotor (x > 0)
        x_mask = x > 0

        # Handle sigma = 0, as this gives 0 in denominator of gaussian's exponent
        # -> gaussian, du, C = 0 when sigma = 0
        sigma_mask = sigma > 0

        comb_mask = x_mask & sigma_mask

        # Gaussian deficit formulation proposed by Bastankhah and Porte-Agel
        # Handle x is a constant 
        if np.shape(x) == () and comb_mask:
            
            # Initialize Gaussian profile and peak deficit
            gaussian = np.zeros(np.shape(y))
            du = np.zeros(np.shape(x))

            du = self._du(x, sigma[comb_mask])

            gaussian[comb_mask] = np.exp(- r[comb_mask] ** 2 /
                            (2 * sigma[comb_mask] ** 2))

        # Handle x is an array
        else:
            
            # Initialize Gaussian profile and peak deficit
            gaussian = np.zeros(np.shape(x))
            du = np.zeros(np.shape(x))

            du[comb_mask] = self._du(x[comb_mask], sigma[comb_mask])

            gaussian[comb_mask] = np.exp(- r[comb_mask] ** 2 /
                            (2 * sigma[comb_mask] ** 2))

        return gaussian * du

    def niayifar_deficit(self, x_glob: ArrayLike, y_glob: ArrayLike, z_glob = 0):
        """
        Computes wake deficit profile, weighted by rotor effective wind speed,
        for use in Niayifar superposition method
        """

        # Calculate deficit
        deficit = self.deficit(x_glob, y_glob, z_glob)

        # Multiply by REWS to get Niayifar deficit
        return self.rotor_sol.REWS * deficit

    def _centerline_wake_added_turb(self, x: ArrayLike):
        """
        Solves Eq. 3 in Pedersen et al. 2022 for I_w(x), turbulence due to 
        wake at the centerline
        """

        I_w = 1 / (self.c_1 + self.c_2 * x/(np.sqrt(self.rotor_sol.Ct / self.rotor_sol.REWS ** 2)))
        I_w[x < 0] = 0.0

        return I_w
    
    def _tot_turb_intensity(self, x: ArrayLike):
        """
        Solves Eq. 2 in Pedersen et al. 2022 for I(x), total turbulence 
        intensity
        """

        I_0 = self.TIamb
        I_w = self._centerline_wake_added_turb(x)
        I_total = np.sqrt(I_0 ** 2 + I_w ** 2)

        return I_total

    def _char_wake_diameter(self, x: ArrayLike) -> ArrayLike:
        """
        Solves Eq. 4 in Pedersen et al. 2022 for nondimensionalized characteristic
        wake diameter sigma. Expression was derived through analytical integration.
        """

        epsilon = self._wake_diameter_prop_const()
        
        # alpha and beta are parameters introduced by Pedersen et al. for 
        # simpler expression evaluation
        alpha = self.c_1 * self.TIamb
        beta = self.c_2 * self.TIamb / np.sqrt(self.rotor_sol.Ct / self.rotor_sol.REWS ** 2)

        # Only compute behind rotor (x > 0)
        x_mask = x > 0

        # Handle log(x) undefined when x <= 0 
        # -> sigma = 0 when log expression undefined
        log_expr = ((np.sqrt((alpha + beta * x) ** 2 + 1) + 1) * alpha) / ((
            np.sqrt(1 + alpha ** 2) + 1) * (alpha + beta * x))
        log_mask = log_expr > 0

        comb_mask = x_mask & log_mask

        sigma = np.zeros(np.shape(x))

        sigma[comb_mask] = epsilon + self.A * self.TIamb / beta * (
                np.sqrt((alpha + beta * x[comb_mask]) ** 2 + 1)
                - np.sqrt(1 + alpha ** 2)
                - np.log(log_expr[comb_mask])
            )

        return sigma

    def _wake_diameter_prop_const(self) -> np.float64:
        """
        Solves Eq. 5 in Pedersen et al. 2022 for epsilon, wake diameter
        at rotor 
        """

        Ct = self.rotor_sol.Ct / self.rotor_sol.REWS ** 2

        epsilon = 0.25 * np.sqrt(
            (1 + np.sqrt(1 - Ct))
            / (2 * np.sqrt(1 - Ct)))

        return epsilon
    
    def _wake_diameter(self, x) -> ArrayLike:
        """
        Compute wake diameter (d) from characteristic wake diameter (sigma)
        using the relation sigma(x) = epsilon * d(x) 
        """

        sigma = self._char_wake_diameter(x)
        epsilon = self._wake_diameter_prop_const()
        
        d = sigma / epsilon

        return d

    def _du(self, x, sigma: Optional[float] = None) -> ArrayLike:
        """
        Solves Eq. 7 in Pedersen et al. for C(x), axial deficit at centerline
        """

        sigma = self._char_wake_diameter(x) if sigma is None else sigma
        
        # Only compute behind rotor (x > 0)
        x_mask = x > 0

        # Handle sigma = 0 nonphysical (there can't be a deficit without a char_wake_diameter)
        # -> du = 0 when sigma <= 0
        sigma_mask = sigma > 0

        # Handle sqrt does not exist -> du does not exist when sqrt does not exist
        sqrt_expr = np.zeros(np.shape(sigma))
        sqrt_expr[sigma_mask] = 1 - (self.rotor_sol.Ct / self.rotor_sol.REWS ** 2) / (8 * sigma[sigma_mask] ** 2)
        sqrt_mask = sqrt_expr >= 0

        comb_mask = x_mask & sigma_mask & sqrt_mask
        
        du = np.zeros(np.shape(sigma))

        du[comb_mask] = 1 - np.sqrt(sqrt_expr[comb_mask])

        return du

    def wake_added_turbulence(
        self, x_glob: ArrayLike, y_glob: ArrayLike, z_glob=0
    ) -> ArrayLike:
        """
        Returns wake added turbulence intensity caused by a wake at particular
        points in space. Laterally smeared with the gaussian twice as wide as
        the wake deficit model. As recommended by Niayifar and Porte-Agel 2016
        """
        x, y, z = x_glob - self.x, y_glob - self.y, z_glob - self.z
        yc = self.centerline(x_glob)[0] - self.y
        zc = self.centerline(x_glob)[1] - self.z

        # Only compute behind rotor (x > 0)
        x_mask = x > 0

        WATI = np.zeros(np.shape(x))
        sigma = np.zeros(np.shape(x))
        d = np.zeros(np.shape(x))
        _gaussian = np.zeros(np.shape(x))

        WATI[x_mask] = self._centerline_wake_added_turb(x[x_mask])
        sigma[x_mask] = self._char_wake_diameter(x[x_mask])
        d[x_mask] = self._wake_diameter(x[x_mask])

        # Handle sigma must be nonzero as this gives 0 in denominator of gaussian's exponent
        # -> gaussian = 0 when sigma = 0
        sigma_mask = sigma != 0
        sigma = np.where(sigma==0, np.nan, sigma)

        comb_mask = x_mask & sigma_mask

        _gaussian[comb_mask] = (
            1
            / (8 * (self.A * sigma[comb_mask]) ** 2)
            * np.exp(
                -(((y[comb_mask] - yc[comb_mask]) ** 2 + (z[comb_mask] - zc[comb_mask])**2)
                    / (2 * (self.A * sigma[comb_mask]) ** 2 * d[comb_mask]**2)
                )
            )
        )

        return _gaussian * np.nan_to_num(WATI)

        # [***FLAG***] Is this same model by Niayifar and Porte Agel applicable for TurbOPark?

    def centerline(self, x_glob: ArrayLike) -> ArrayLike:
        """
        No wake deflection implemented for TurbOPark wake model, as
        Shapiro et al. 2018 method is likely to overpredict deflection        
        """

        x = x_glob - self.x

        # Zero wake deflection for no yaw, no tilt case
        if (self.rotor_sol.yaw != 0) or (self.rotor_sol.tilt != 0):
            raise NotImplementedError("No wake deflection implemented for Gaussian TurbOPark wake model.")
        else: 
            yc = np.full(np.shape(x), self.y)
            zc = np.full(np.shape(x), self.z) 

        return yc, zc
    

class TopHatTurbOParkWake(Wake):
    """
    Attributes:
        x: x-position of rotor in global coordinate frame
        y: y-position of rotor in global coordinate frame
        z: z-position of rotor in global coordinate frame
        rotor_sol: Rotor solution
        TIamb: Ambient turbulence intensity
        A: Wake expansion calibration parameter 
        c_1, c_2: Empirically estimated constants in wake-added turbulence intensity
        xmax: Maximum x value evaluated
        dx: Interval of x values evaluated
    """

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        rotor_sol: "RotorSolution",
        TIamb: float = 0.1,
        A: float = 0.6,
        c_1: float = 1.5,
        c_2: float = 0.8,
        xmax: float = 100.0,
        dx: float = 0.05
    ):
        self.x, self.y, self.z = x, y, z
        self.rotor_sol = rotor_sol
        self.TIamb = TIamb
        self.A = A
        self.c_1 = c_1
        self.c_2 = c_2

    def deficit(self, x_glob: ArrayLike, y_glob: ArrayLike, z_glob = 0):
        """
        Solves Eq. 6 in Pedersen et al. 2022 for wake deficit profile
        """

        x = np.float64(x_glob - self.x)
        y = np.float64(y_glob - self.y)
        z = np.float64(z_glob - self.z)
        yc = self.centerline(x_glob)[0] - self.y
        zc = self.centerline(x_glob)[1] - self.z
        r = np.sqrt((y - yc) ** 2 + (z - zc) ** 2) 

        d = self._wake_diameter(x)

        # Only compute for x > 0
        x_mask = x > 0

        du = np.zeros(np.shape(d))

        du[x_mask] = self._du(x[x_mask], d[x_mask])

        # Set deficit to be only inside wake region
        # Handle x is a constant
        if np.shape(x) == ():
            deficit = np.zeros(np.shape(y))

            for ii, _ in np.ndenumerate(deficit):
                if (np.sqrt(r[ii] ** 2) <= 1/2*d) and (x >= 0):
                    deficit[ii] = du
                else:
                    # Outside of wake, there is no deficit (velocity is the same as u_inf)
                    deficit[ii] = 0

        # Handle x is an array
        else:
            deficit = np.zeros(np.shape(x))

            for ii, _ in np.ndenumerate(deficit):
                if (np.sqrt(r[ii] ** 2) <= 1/2*d[ii]) and (x[ii] >= 0):
                    deficit[ii] = du[ii]
                else:
                    # Outside of wake, there is no deficit (velocity is the same as u_inf)
                    deficit[ii] = 0

        return deficit

    def niayifar_deficit(self, x_glob: ArrayLike, y_glob: ArrayLike, z_glob = 0):
        """
        Computes wake deficit profile, weighted by rotor effective wind speed,
        for use in Niayifar superposition method
        """

        # Calculate deficit 
        deficit = self.deficit(x_glob, y_glob, z_glob)

        return deficit * self.rotor_sol.REWS

    def _centerline_wake_added_turb(self, x: ArrayLike):
        """
        Solves Eq. 3 in Pedersen et al. 2022 for I_w(x), turbulence due to 
        wake at the centerline
        """

        I_w = 1 / (self.c_1 + self.c_2 * x/(np.sqrt(self.rotor_sol.Ct / self.rotor_sol.REWS ** 2)))
        I_w[x < 0] = 0.0

        return I_w
    
    def _tot_turb_intensity(self, x: ArrayLike):
        """
        Solves Eq. 2 in Pedersen et al. 2022 for I(x), total turbulence 
        intensity
        """

        I_0 = self.TIamb
        I_w = self._centerline_wake_added_turb(x)
        I_total = np.sqrt(I_0 ** 2 + I_w ** 2)

        return I_total
    
    def _wake_diameter(self, x) -> ArrayLike:
        """
        Compute wake diameter (d) from characteristic wake diameter (sigma)
        using the relation sigma(x) = epsilon * d(x) 
        """

        # alpha and beta are parameters introduced by Nygaard et al. for 
        # simpler expression evaluation
        alpha = self.c_1 * self.TIamb
        # beta = self.c_2 * self.TIamb / np.sqrt(self.rotor_sol.Ct)
        beta = self.c_2 * self.TIamb / np.sqrt(self.rotor_sol.Ct / self.rotor_sol.REWS ** 2)

        # Only compute behind rotor (x > 0)
        x_mask = x > 0

        # Handle log(x) undefined when x <= 0 
        # -> d = 0 when log expression undefined
        log_expr = ((np.sqrt((alpha + beta * x) ** 2 + 1) + 1) * alpha) / ((
            np.sqrt(1 + alpha ** 2) + 1) * (alpha + beta * x))
        log_mask = log_expr > 0

        comb_mask = x_mask & log_mask

        d = np.zeros(np.shape(x))

        d[comb_mask] = 1 + self.A * self.TIamb / beta * (
            np.sqrt((alpha + beta * x[comb_mask]) ** 2 + 1)
            - np.sqrt(1 + alpha ** 2)
            - np.log(log_expr[comb_mask])
        )

        return d

    def _du(self, x, wake_diameter: Optional[float] = None) -> ArrayLike:
        """
        Solves Eq. 7 in Pedersen et al. for C(x), deficit at centerline
        """

        d = self._wake_diameter(x) if wake_diameter is None else wake_diameter
        
        # Only compute for x > 0
        x_mask = x > 0

        # Handle d = 0 -> du = 0 when sigma = 0
        d_mask = d > 0

        comb_mask = x_mask & d_mask
        
        du = np.zeros(np.shape(d))

        # du[comb_mask] = (1 - self.rotor_sol.REWS * np.sqrt(1 - self.rotor_sol.Ct)) / (d[comb_mask]**2)
        du[comb_mask] = (1 - self.rotor_sol.REWS * np.sqrt(1 - self.rotor_sol.Ct / self.rotor_sol.REWS ** 2)) / (d[comb_mask]**2)

        return du

    def wake_added_turbulence(
        self, x_glob: ArrayLike, y_glob: ArrayLike, z_glob=0
    ) -> ArrayLike:

        # Placeholder of zeroes
        return np.zeros(np.shape(x_glob))

    def centerline(self, x_glob: ArrayLike) -> ArrayLike:
        """
        No wake deflection implemented for TurbOPark wake model, as
        Shapiro et al. 2018 method is likely to overpredict deflection        
        """

        x = x_glob - self.x

        # Zero wake deflection for no yaw, no tilt case
        if (self.rotor_sol.yaw != 0) or (self.rotor_sol.tilt != 0):
            raise NotImplementedError("No wake deflection implemented for top hat TurbOPark wake model.")
        else: 
            yc = np.full(np.shape(x), self.y)
            zc = np.full(np.shape(x), self.z) 

        return yc, zc


class TurbOParkWakeModel(WakeModel):
    def __init__(
        self,
        A: float = 0.04,
        c_1: float = 1.5,
        c_2: float = 0.8,
        xmax: float = 100.0,
        dx: float = 0.05,
        gaussian_profile: bool = True
    ):
        self.xmax = xmax
        self.dx = dx
        self.A = A
        self.c_1 = c_1
        self.c_2 = c_2
        self.gaussian_profile = gaussian_profile

    def __call__(
        self,
        x,
        y,
        z, 
        rotor_sol: "RotorSolution",
        TIamb: float = 0.06
    ) -> TurbOParkWake:
        
        # Return Gaussian TurbOPark by default (Pedersen et al. 2022)
        if self.gaussian_profile == True:
            return TurbOParkWake(
                x,
                y,
                z,
                rotor_sol,
                TIamb = TIamb,
                xmax = self.xmax,
                dx = self.dx,
                A = self.A,
                c_1 = self.c_1,
                c_2 = self.c_2,
            )
        
        # Return Top Hat TurbOPark if desired (Nygaard et al. 2020)
        else:
            return TopHatTurbOParkWake(
                x,
                y,
                z,
                rotor_sol,
                TIamb = TIamb,
                xmax = self.xmax,
                dx = self.dx,
                A = self.A,
                c_1 = self.c_1,
                c_2 = self.c_2,
            )


class GaussianWake(Wake):
    """
    Gaussian wake model using the derivation from Shapiro et al. (2018)
    with a wake added turbulence model from Crespo and Hernandez (1996).

    The model uses a constant wake spreading rate (kw) and a constant standard
    deviation (sigma) for the Gaussian profile. 

    __init__: 
        - Args
            - sigma: float, standard deviation of the Gaussian profile (default: 0.25)
            - kw: float, wake spreading rate (default: 0.07)
            - WATI_sigma_multiplier: float, multiplier for the wake added 
                turbulence intensity (WATI) sigma (default: 1.0)
            - xmax: float, maximum x value for the centerline calculation (default: 100.0)

    __call__: function to create a GaussianWake instance called by the wake model solver
        - Args
            - x: float, x-coordinate of the turbine
            - y: float, y-coordinate of the turbine
            - z: float, z-coordinate of the turbine
            - rotor_sol: RotorSolution, solution object containing rotor parameters
            - TIamb: float, ambient turbulence intensity (default: None)
        - Returns:
            - GaussianWake instance with the specified parameters.
    """
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
        self.x_centerline, self.yz_centerline = self._centerline(xmax, dx)

    def __repr__(self):
        return f"GaussianWake(x={self.x}, y={self.y}, z={self.z}, sigma={self.sigma}, kw={self.kw})"

    def _centerline(self, xmax: float, dx: float = 0.05) -> ArrayLike:
        """
        Returns centerline y/z position in global coordinates without constant v4/w4 factor.
        Eqs. C3 and C4 in Heck et al 2023 Appendix C / Eqs. 3.8-3.9 in Shapiro et al 2018.
        """
        _x = np.arange(0, max(xmax, 2 * dx), dx)
        d = self._wake_diameter(_x)
        # v and w deficit values at _x, without constant factors of v4 and w4 (see equation C3)
        dvw = -0.5 / d**2 * (1 + erf(_x / (np.sqrt(2) / 2)))
        # y and z centerline without  without constant factors of v4 and w4 (see integration in C4)
        _yzc = cumulative_trapezoid(-dvw, dx=dx, initial=0)

        return _x, _yzc

    def centerline(self, x_glob: ArrayLike) -> ArrayLike:
        """
        Returns centerline y position in global coordinates - scales and translates the results of the `_centerline` function.
        Eq. C4. in Heck et al 2023 Appendix C / Eq. 3.9 in Shapiro et al 2018.
        """
        x = x_glob - self.x
        yzc_temp = np.interp(x, self.x_centerline, self.yz_centerline, left=0)
        # scale and translate centerlines
        yc = yzc_temp * self.rotor_sol.v4 + self.y
        zc = yzc_temp * self.rotor_sol.w4 + self.z
        return yc, zc

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
        Solves the normalized far-wake diameter. Note that this is non-dimensionalized by D and x is actually x/D.
        Defined in text between Eq. C1 and C2 in Heck et al 2023 Appendix C / in text between Eq. 3.4-3.5 in Shapiro et al 2018.

        Note that this is non-dimensionalized by D and x is actually x/D.
        """
        return 1 + self.kw * np.log(1 + np.exp(2 * (x - 1)))

    def _du(self, x: ArrayLike, wake_diameter: Optional[float] = None) -> ArrayLike:
        """
        Calcualtes the streamwise velocity deficit.
        Eq. C2. in Heck et al 2023 Appendix C / Eq. 3.7 in Shapiro et al 2018.
        """
        d = self._wake_diameter(x) if wake_diameter is None else wake_diameter

        du = 0.5 * (1 - (self.rotor_sol.u4 / self.rotor_sol.REWS)) / d**2 * (1 + erf(x / (np.sqrt(2) / 2)))
        return du

    def _gaussian(self, x_glob, y_glob, z_glob, sigma_multiplier = 1):
        """
        Defines a Gaussian profile centered at the turbine.
        Gaussian portion of Eq. C1 in Heck et al 2023 Appendix C / Eq. 3.5 in Shapiro et al 2018 (without streamwise velocity deficit multiplier).
        """
        # calculate the centerline
        yc, zc = self.centerline(x_glob)
        # transform coordinates to be in turbine/wake centerline frame of reference
        x = x_glob - self.x
        y = y_glob - yc
        z = z_glob - zc
        # find wake diameter
        d = self._wake_diameter(x)
        # calculate sigma
        sigma = self.sigma * sigma_multiplier
        # calculate gaussian
        gaussian_ = (
            1
            / (8 * sigma**2)
            * np.exp(-((y** 2 + z**2) / (2 * sigma**2 * d**2)))
        )
        return gaussian_, x, d


    def deficit(self, x_glob: ArrayLike, y_glob: ArrayLike, z_glob=0) -> ArrayLike:
        """
        Distributes average streamwise velocity deficit using a Gaussian profile centered at the turbine.
        Eq. C1 in Heck et al 2023 Appendix C / Eq. 3.5 in Shapiro et al 2018.
        """
        gaussian_, x, d = self._gaussian(x_glob, y_glob, z_glob)
        du = self._du(x, wake_diameter=d)
        return gaussian_ * du

    def wake_added_turbulence(
        self, x_glob: ArrayLike, y_glob: ArrayLike, z_glob=0
    ) -> ArrayLike:
        """
        Returns wake added turbulence intensity caused by a wake at particular
        points in space. Laterally smeared with the gaussian twice as wide as
        the wake deficit model. As recommended by Niayifar and Porte-Agel 2016
        """
        gaussian_, x, _ = self._gaussian(x_glob, y_glob, z_glob, sigma_multiplier = self.WATI_sigma_multiplier)
        WATI = self.centerline_wake_added_turb(x)
        return gaussian_ * np.nan_to_num(WATI)

    def line_deficit(self, x: np.array, y: np.array, z = 0):
        """
        Returns the deficit at hub height averaged along a lateral line of
        length 1, centered at (x, y).
        """
        warnings.warn("Line deficit is deprecated as it cannot handle a z-offset. Use another deficit function.", DeprecationWarning)
        d = self._wake_diameter(x)
        yc, zc = self.centerline(x)
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