"""
Unified Momentum Model Rotor Definitions

This module defines classes representing different rotor models based on the Unified Momentum Model.
It includes abstract classes and concrete implementations such as BEM, UnifiedAD, and AD.

Classes:
- Rotor: Abstract base class for rotor models.
- BEM: Blade Element Momentum (BEM) rotor model.
- UnifiedAD: Unified Momentum Model actuator disk.
- AD: Axial Distribution rotor model.

Data Classes:
- RotorSolution: Data class representing the solution of rotor models.

Usage Example:
    rotor_def = RotorDefinition(...)  # Define rotor parameters
    bem_rotor = BEM(rotor_def)         # Create a BEM rotor instance
    solution = bem_rotor(pitch, tsr, yaw, tilt)  # Calculate rotor solution for given inputs
    print(solution.Cp, solution.Ct, solution.Ctprime, solution.an, solution.u4, solution.v4)

Note: Make sure to replace '...' with the actual parameters in RotorDefinition.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from numpy.typing import ArrayLike
from typing import Tuple
import warnings

import numpy as np
from scipy.optimize import root
from UnifiedMomentumModel.Momentum import Heck, UnifiedMomentum, MomentumSolution
from MITRotor import BEM as _BEM
from MITRotor import BEMSolution, RotorDefinition
from .Windfield import Windfield
from .RotorGrid import RotorGrid, Point, Line, Area


@dataclass
class RotorSolution:
    """
    Data class representing the solution of rotor models.

    Note that non-dimensional values are returned by the rotors and the values are
    dimensionalized by being multipled by the needed factor of REWS.
    """

    yaw: float
    Cp: float
    Ct: float
    Ctprime: float
    an: float
    u4: float
    v4: float
    REWS: float
    # optional keywords
    tilt: float = 0
    w4: float = 0
    TI: float = None
    idx: int = None
    extra: Any = None


class Rotor(ABC):
    """
    Abstract base class for rotor models.

    Subclasses must implement the __call__ method.
    """

    @abstractmethod
    def __call__(self, *args) -> RotorSolution:
        """
        Calculate the rotor solution for given input parameters.

        Parameters:
        - args: Input parameters specific to the rotor model.

        Returns:
        RotorSolution: The calculated rotor solution.
        """
        pass


class AD(Rotor):
    """
    Axial Distribution rotor model.

    __init__:
        - Args:
            - rotor_grid (RotorGrid, optional): grid points over the rotor
        - Returns: AD object
        - Example:
            >>> rotor_model = AD()

    __call__:
        - Args:
            - Ctprime (float): Thrust coefficient including the effect of yaw and tilt.
            - yaw (float, optional): Yaw angle of the rotor.
            - tilt (float, optional): Tilt angle of the rotor
        - Returns: RotorSolution calculted by the Heck momentum model with high thrust corrrection given arguments
        - Example:
            >>> rotor_model(1.33, np.deg2rad(15), 0)
    """

    def __init__(self, rotor_grid: RotorGrid = None):
        """
        Initialize the AD rotor model using the Heck momentum model.
        See above class documentation on __init__ for more details.
        """
        self._model = Heck()
        if rotor_grid is None:
            self.rotor_grid = Area()
        else:
            self.rotor_grid = rotor_grid

    def __call__(self, x: float, y: float, z: float, windfield: Windfield, Ctprime, yaw = 0, tilt = 0) -> RotorSolution:
        """
        Calculate the rotor solution using the Heck momentum model for given Ctprime, yaw, and tilt inputs.
        See above class documentation on __call__ for more details.
        """
        # Calculate rotor solution (independent of wind field in this model)
        sol: MomentumSolution = self._model(Ctprime, yaw = yaw, tilt = tilt)

        # Get the points over rotor to be sampled in windfield
        xs_loc, ys_loc, zs_loc = self.rotor_grid.grid_points()
        xs_glob, ys_glob, zs_glob = xs_loc + x, ys_loc + y, zs_loc + z

        # sample windfield and calculate rotor effective wind speed
        Us = windfield.wsp(xs_glob, ys_glob, zs_glob)
        TIs = windfield.TI(xs_glob, ys_glob, zs_glob)
        
        REWS = self.rotor_grid.average(Us)
        RETI = np.sqrt(self.rotor_grid.average(TIs**2))

        # rotor solution is normalised by REWS. Convert normalisation to U_inf and return
        return RotorSolution(
            yaw,
            sol.Cp * REWS**3,
            sol.Ct * REWS**2,
            sol.Ctprime,
            sol.an * REWS,
            sol.u4 * REWS,
            sol.v4 * REWS,
            REWS,
            tilt = tilt,
            w4 = sol.w4 * REWS,
            TI=RETI,
            extra=sol,
        )


class UnifiedAD(Rotor):
    """
    Unified Momentum Model rotor.

    __init__:
        - Args:
            - rotor_grid (RotorGrid, optional): grid points over the rotor
            - beta_s (float, optional): shear layer growth parameter.
                Default is 0.1403 (from Liew et al. 2024).
        - Returns:
            - UnifiedAD object
        - Example:
            >>> rotor_model = UnifiedAD()

    __call__:
        - Args:
            - Ctprime (float): Thrust coefficient including the effect of yaw and tilt.
            - yaw (float, optional): Yaw angle of the rotor.
            - tilt (float, optional): Tilt angle of the rotor
        - Returns: RotorSolution calculted by the Unified Momentum model given arguments
        - Example:
            >>> rotor_model(1.33, 0, np.deg2rad(-15))
    """

    def __init__(self, rotor_grid: RotorGrid = None, beta_s=0.1403):
        """
        Initialize the UnifiedAD rotor model.
        See above class documentation on __init__ for more details.
        """
        if rotor_grid is None:
            self.rotor_grid = Point()
        else:
            self.rotor_grid = rotor_grid
        self._model = UnifiedMomentum(beta_s=beta_s)

    def __call__(self, x: float, y: float, z: float, windfield: Windfield, Ctprime, yaw = 0, tilt = 0) -> RotorSolution:
        """
        Calculate the rotor solution using the Unified Momentum Model for given Ctprime, yaw, and tilt inputs.
        See above class documentation on __call__ for more details.
        """

        # Get the points over rotor to be sampled in windfield
        xs_loc, ys_loc, zs_loc = self.rotor_grid.grid_points()
        xs_glob, ys_glob, zs_glob = xs_loc + x, ys_loc + y, zs_loc + z

        # sample windfield and calculate rotor effective wind speed
        Us = windfield.wsp(xs_glob, ys_glob, zs_glob)
        TIs = windfield.TI(xs_glob, ys_glob, zs_glob)

        REWS = self.rotor_grid.average(Us)
        RETI = np.sqrt(self.rotor_grid.average(TIs**2))
        sol: MomentumSolution = self._model(Ctprime, yaw = yaw, tilt = tilt, TI=RETI)

        # rotor solution is normalised by REWS. Convert normalisation to U_inf and return
        return RotorSolution(
            yaw,
            sol.Cp * REWS**3,
            sol.Ct * REWS**2,
            sol.Ctprime,
            sol.an * REWS,
            sol.u4 * REWS,
            sol.v4 * REWS,
            REWS,
            tilt = tilt,
            w4 = sol.w4 * REWS,
            TI=RETI,
            extra=sol,
        )


class UnifiedAD_TI(UnifiedAD):
    """
    Same as UnifiedAD but also accounts for a possible TI dependence

    Inherits the __call__ function from UnifiedAD. 

    __init__:
        - Args:
            - rotor_grid (RotorGrid, optional): grid points over the rotor
            - beta (float, optional): shear layer growth parameter. Default is 0.1403.
            - alpha (float, optional): turbulence intensity factor, default: 2.32 
                This is alpha^* in Bastankhah and Porté-Agel (2016).
            - couple_x0 (bool, optional): If True, couples the x0 parameter to the
                pressure solver. Default is False. (Coupling not recommended)
        - Returns:
            - UnifiedAD_TI object
        - Example:
            >>> rotor_model = UnifiedAD()

    """

    def __init__(self, rotor_grid=None, beta_s=0.1403, alpha=2.32, couple_x0=False):
        """
        Initialize the UnifiedAD rotor model given shear layer growth parameters alpha, beta.

        Parameters:
        - beta_s (float): shear layer growth parameter (default is 0.1403, from Liew et al. 2024).
        - alpha (float): Turbulence intensity factor (default is 2.32, from Bastankhah and Porté-Agel 2016).
        - couple_x0 (bool): If True, couples the x0 parameter to the pressure equation. Default is False.
        """
        super().__init__(rotor_grid=rotor_grid)
        if couple_x0:
            self._model = UnifiedMomentumTI(beta_s=beta_s, alpha=alpha)
        else:
            self._model = UnifiedMomentumTI_x0(beta_s=beta_s, alpha=alpha)


class BEM(Rotor):
    """
    Blade Element Momentum (BEM) rotor model. Note: MITRotor is formulated in
    terms of rotor radius, whereas MITWindfarm is in rotor diameters.
    Conversions MUST be made between the two normalizations in this class.

    __init__:
        - Args:
            - rotor_definition (RotorDefinition): Definition of the rotor parameters.
            - BEM_model (BEMModel, optional): BEM Model (potentially a user-defined model) that will be used rather than the default BEM from MITRotor
            - **kwargs: Additional keyword arguments passed to the underlying BEM model.
        - Returns:
            - BEM objects

    __call__:
        - Args:
            - x (float): x location of rotor
            - y (float): y location of rotor
            - z (float): z location of rotor
            - windfield (Windfield): windfield in simulation as 
            - pitch (float): Pitch angle of the rotor blades.
            - tsr (float): Tip-speed ratio of the rotor.
            - yaw (float): Yaw angle of the rotor.
        - Returns:
            - RotorSolution with calculated BEM solution based on arguments.
    """

    def __init__(self, rotor_definition: RotorDefinition, BEM_model=None, **kwargs):
        """
        Initialize the BEM rotor model with the given rotor definition.
        See above class documentation on __init__ for more details.
        """
        BEM_model = BEM_model or _BEM
        self._model = BEM_model(rotor_definition, **kwargs)
        self.xgrid_loc, self.ygrid_loc, self.zgrid_loc = self._model.sample_points()
        # Convert from radius to diameter normalization
        self.xgrid_loc /= 2
        self.ygrid_loc /= 2
        self.zgrid_loc /= 2

    def __call__(self, x: float, y: float, z: float, windfield: Windfield, pitch, tsr, yaw = 0, tilt = 0) -> RotorSolution:
        """
        Calculate the RotorSolution for given pitch, TSR, and yaw inputs.
        See above class documentation on __call__ for more details.
        """
        xs_glob = self.xgrid_loc + x
        ys_glob = self.ygrid_loc + y
        zs_glob = self.zgrid_loc + z
        Us = windfield.wsp(xs_glob, ys_glob, zs_glob)
        TIs = windfield.TI(xs_glob, ys_glob, zs_glob)

        REWS = self._model.geometry.rotor_average(self._model.geometry.annulus_average(Us))
        RETI = np.sqrt(self._model.geometry.rotor_average(self._model.geometry.annulus_average(TIs**2)))

        wdir = windfield.wdir(xs_glob, ys_glob, zs_glob)
        sol: BEMSolution = self._model(pitch, tsr, yaw = yaw, tilt = tilt, U = Us / REWS, wdir = wdir)
        return RotorSolution(
            yaw,
            sol.Cp() * REWS**3,
            sol.Ct() * REWS**2,
            sol.Ctprime(),
            sol.a() * REWS,
            sol.u4 * REWS,
            sol.v4 * REWS,
            REWS,
            tilt = tilt,
            w4 = sol.w4 * REWS,
            TI=RETI,
            extra=sol,
        )
    
class CosineRotor(Rotor):
    """
    __init__:
        - Args:
            - windspeeds_over_urated (array): Array of wind speeds normalized by rated wind speed.
            - Cts (array): Array of thrust coefficients.
            - Cps (array): Array of power coefficients.
            - Pp (float): Power cosine exponent.
            - Tp (float): Thrust cosine exponent.
            - urated_over_freestream (float): Rated wind speed normalized by freestream wind speed.
        - Returns: CosineRotor
        - Example:
            >>>

    __call__:
        - Args:
            - x (float): x location of rotor
            - y (float): y location of rotor
            - z (float): z location of rotor
            - windfield (Windfield): windfield in simulation as 
            - yaw (float): Yaw angle of the rotor.
        - Returns: RotorSolution
        - Example:
            >>>
    """
    def __init__(self, 
                 windspeeds_over_urated: ArrayLike, 
                 Cts: ArrayLike, 
                 Cps: ArrayLike, 
                 Pp: float, 
                 Tp: float, 
                 urated_over_freestream: float,
                 rotor_grid: RotorGrid = None):
        """
        Initialize the CosineRotor model with given wind speeds and coefficients.
        See above class documentation on __init__ for more details.
        """

        self.windspeeds_over_urated = windspeeds_over_urated
        self.Cts = Cts
        self.Cps = Cps
        self.Pp = Pp
        self.Tp = Tp
        self.urated_over_freestream = urated_over_freestream
        self.windspeeds_over_freestream = windspeeds_over_urated * urated_over_freestream
        if rotor_grid is None:
            self.rotor_grid = Point()
        else:
            self.rotor_grid = rotor_grid

    def compute_initial_wake_velocities(self, Ct: float, yaw: float) -> float:
        a = 0.5 * (1 - np.sqrt(1 - Ct))
        u4 = np.sqrt(1 - Ct)
        v4 = - (1/4) * Ct * np.sin(yaw)
        return a, u4, v4

    def __call__(self, x: float, y: float, z: float, windfield: Windfield, yaw = 0, tilt = 0) -> RotorSolution:
        """
        Calculate the rotor solution for the cosine rotor.
        See above class documentation on __call__ for more details.
        """
        if tilt != 0:
            warnings.warn("Non-zero tilt is not yet implemented for Cosine rotors. Setting tilt to zero.", UserWarning)
            tilt = 0

        # Get the points over rotor to be sampled in windfield
        xs_loc, ys_loc, zs_loc = self.rotor_grid.grid_points()
        xs_glob, ys_glob, zs_glob = xs_loc + x, ys_loc + y, zs_loc + z

        # sample windfield and calculate rotor effective wind speed
        Us = windfield.wsp(xs_glob, ys_glob, zs_glob)
        TIs = windfield.TI(xs_glob, ys_glob, zs_glob)

        REWS = self.rotor_grid.average(Us)
        RETI = np.sqrt(self.rotor_grid.average(TIs**2))


        # Interpolate thrust and power coefficients based on wind speed
        Ct_y0 = np.interp(REWS, self.windspeeds_over_freestream, self.Cts)
        Cp_y0 = np.interp(REWS, self.windspeeds_over_freestream, self.Cps)

        # Calculate thrust coefficient with cosine correction
        Ct = Ct_y0 * np.cos(np.radians(yaw))**self.Tp

        # Calculate power coefficient with cosine correction
        Cp = Cp_y0 * np.cos(np.radians(yaw))**self.Pp

        # evaluate classical induction model
        a, u4, v4 = self.compute_initial_wake_velocities(Ct, yaw)

        return RotorSolution(
            yaw,
            Cp * REWS**3,
            Ct * REWS**2,
            np.nan,
            a * REWS,
            u4 * REWS,
            v4 * REWS,
            REWS,
            TI=RETI,
            extra=None
        )


# Custom momentum models - move to UnifiedMomentum later on
class UnifiedMomentumTI_x0(UnifiedMomentum):
    """
    Here, the influence of TI on x0 is decoupled from the other near-wake equations.

    Inherits __call__, initial_guess, residual, and pre_process from UnifiedMomentum. 

    __init__:
        - Args:
            - beta (float, optional): shear layer growth parameter. Default is 0.1403.
            - alpha (float, optional): turbulence intensity factor. Default is 2.32
                This is alpha^* in Bastankhah and Porté-Agel (2016).
            cached (bool, optional): Caches the pressure solver. Default is True.
            - v4_correction (float, optional): Correction factor for the lateral outlet velocity.
                Default is 1.0.
        - Returns:
            - UnifiedMomentumTI_x0 object
        - Example:
            >>> momentum = UnifiedMomentumTI_x0()
            >>> momentum_sol = momentum(Ctprime, yaw, TI=0.05)
    """

    def __init__(
        self, beta_s=0.1403, alpha=2.32, cached=True, v4_correction=1.0, **kwargs
    ):
        super().__init__(
            beta_s=beta_s, cached=cached, v4_correction=v4_correction, **kwargs
        )
        self.alpha = alpha

    def post_process(self, result, Ctprime, yaw = 0, tilt = 0, TI = 0, **kwargs):
        a, u4, v4, _x0, dp = result.x
        x0 = (
            np.cos(self.eff_yaw)
            / 4
            * (1 + u4)
            * np.sqrt((1 - a) * np.cos(self.eff_yaw) / (1 + u4))
            / (self.beta_s * np.abs(1 - u4) / 2 + self.alpha * TI)
        )  # re-compute x0 with TI influence decoupled
        result.x = (a, u4, v4, x0, dp)
        return super().post_process(result, Ctprime, yaw = yaw, tilt = tilt, **kwargs)


class UnifiedMomentumTI(UnifiedMomentum):
    """
    Extends the Unified Momentum Model to include a TI dependence
    as described in Bastankhah and Porté-Agel (2016).

    Here, the influence of TI on x0 is coupled to the other near-wake equations.
    (not recommended for use)

    __init__:
        - Args:
            - beta_s (float, optional): shear layer growth parameter. Default is 0.1403.
            - alpha (float, optional): turbulence intensity factor. Default is 2.32
                This is alpha^* in Bastankhah and Porté-Agel (2016).
            cached (bool, optional): Caches the pressure solver. Default is True.
            - v4_correction (float, optional): Correction factor for the lateral outlet velocity.
                Default is 1.0.
        - Returns:
            - UnifiedMomentumTI object
        - Example:
            >>> momentum = UnifiedMomentumTI()
            >>> momentum_sol = momentum(Ctprime, yaw, TI=0.05)

    __call__: 
        - Args:
            - Ctprime (float): Thrust coefficient including the effect of yaw and tilt.
            - yaw (float, optional): Yaw angle of the rotor in radians
            - tilt (float, optional): Tilt angle of the rotor in radians
            - TI (float, optional): Turbulence intensity at the rotor location.
        - Returns: MomentumSolution with calculated Unified Momentum solution based on arguments
        - Example:
            >>> momentum = UnifiedMomentumTI()
            >>> momentum_sol = momentum(Ctprime, yaw=0, TI=0.05)
    """
    def __init__(self, beta_s=0.1403, alpha=2.32, **kwargs):
        super().__init__(beta_s=beta_s, **kwargs)
        self.alpha = alpha

    def residual(
        self, x: np.ndarray, Ctprime: float, TI: float = 0, **kwargs
    ) -> Tuple[float, ...]:
        """
        Returns the residuals of the Unified Momentum Model for the fixed point
        iteration. The equations referred to in this function are from the
        associated paper.
        """
        an, u4, v4, x0, dp = x
        if type(Ctprime) is float and Ctprime == 0:
            return 0 - an, 1 - u4, 0 - v4, 100 - x0, 0 - dp

        p_g = self._nonlinear_pressure(Ctprime, self.eff_yaw, an, x0)

        # Eq. 4 - Near wake length in residual form, includes alpha term.
        e_x0 = (
            np.cos(self.eff_yaw)
            / 4
            * (1 + u4)
            * np.sqrt((1 - an) * np.cos(self.eff_yaw) / (1 + u4))
            / (self.beta_s * np.abs(1 - u4) / 2 + self.alpha * TI)
        ) - x0

        # Eq. 1 - Rotor-normal induction in residual form.
        e_an = (
            1
            - np.sqrt(
                -dp / (0.5 * Ctprime * np.cos(self.eff_yaw) ** 2)
                + (1 - u4**2 - v4**2) / (Ctprime * np.cos(self.eff_yaw) ** 2)
            )
        ) - an

        # Eq. 2 - Streamwise outlet velocity in residual form.
        e_u4 = (
            -(1 / 4) * Ctprime * (1 - an) * np.cos(self.eff_yaw) ** 2
            + (1 / 2)
            + (1 / 2)
            * np.sqrt(
                (1 / 2 * Ctprime * (1 - an) * np.cos(self.eff_yaw) ** 2 - 1) ** 2 - (4 * dp)
            )
        ) - u4

        # Eq. 3 - Lateral outlet velocity in residual form.
        e_v4 = (
            -self.v4_correction
            * (1 / 4)
            * Ctprime
            * (1 - an) ** 2
            * np.sin(self.eff_yaw)
            * np.cos(self.eff_yaw) ** 2
            - v4
        )

        # Eq. 5 - Outlet pressure drop in residual form.
        e_dp = (
            (
                -(1 / (2 * np.pi))
                * Ctprime
                * (1 - an) ** 2
                * np.cos(self.eff_yaw) ** 2
                * np.arctan(1 / (2 * x0))
            )
            + p_g
        ) - dp

        return e_an, e_u4, e_v4, e_x0, e_dp


class UnifiedAD_veer(UnifiedAD):
    """
    Same as UnifiedAD but also accounts for a possible dependence on veer and inflow TI. 
    """

    def __init__(self, rotor_grid=None, beta_s=0.1403, alpha=2.32):
        """
        Initialize the UnifiedAD rotor model.

        Parameters:
        - beta_s (float): shear layer growth parameter (default is 0.1403).
        - alpha (float): Turbulence intensity factor (default is 2.32, 
            which is alpha^* from Bastankhah and Porté-Agel 2016).
        """
        super().__init__(rotor_grid=rotor_grid)
        self._model = UnifiedMomentum_veer(beta_s=beta_s, alpha=alpha)

    def __call__(
        self, x: float, y: float, z: float, windfield: Windfield, Ctprime, yaw=0, tilt=0
    ) -> RotorSolution:
        """
        Calculate the rotor solution for given Ctprime and yaw inputs.

        Parameters:
        - Ctprime (float): Thrust coefficient including the effect of yaw.
        - yaw (float, optional): Yaw angle of the rotor, in radians
        - tilt (float, optional): Tilt angle of the rotor, in radians

        Returns:
        RotorSolution: The calculated rotor solution.
        """

        # Get the points over rotor to be sampled in windfield
        xs_loc, ys_loc, zs_loc = self.rotor_grid.grid_points()
        xs_glob, ys_glob, zs_glob = xs_loc + x, ys_loc + y, zs_loc + z

        # sample windfield and calculate rotor effective wind speed
        Us = windfield.wsp(xs_glob, ys_glob, zs_glob)
        TIs = windfield.TI(xs_glob, ys_glob, zs_glob)

        REWS = self.rotor_grid.average(Us)
        RETI = np.sqrt(self.rotor_grid.average(TIs**2))

        # compute veer; this sampling should be done in a vertical line at the rotor: 
        zax = np.linspace(-0.5, 0.5) + z
        xax = np.full_like(zax, x)
        yax = np.full_like(zax, y)
        veer = -np.mean(np.gradient(windfield.wdir(xax, yax, zax), zax))

        sol = self._model(Ctprime, yaw, TI=RETI, veer=veer)

        # rotor solution is normalised by REWS. Convert normalisation to U_inf and return
        return RotorSolution(
            yaw,
            sol.Cp * REWS**3,
            sol.Ct * REWS**2,
            sol.Ctprime,
            sol.an * REWS,
            sol.u4 * REWS,
            sol.v4 * REWS,
            REWS,
            TI=RETI,
            extra=sol,
        )
    

class UnifiedMomentum_veer(UnifiedMomentum):
    """
    Here, the influence of TI on x0 is decoupled from the other near-wake equations.

    Inherits __call__, initial_guess, residual, and pre_process from UnifiedMomentum. 

    __init__:
        - Args:
            - beta_s (float, optional): shear layer growth parameter. Default is 0.1403.
            - alpha (float, optional): turbulence intensity factor. Default is 2.32
                This is alpha^* in Bastankhah and Porté-Agel (2016).
            cached (bool, optional): Caches the pressure solver. Default is True.
            - v4_correction (float, optional): Correction factor for the lateral outlet velocity.
                Default is 1.0.
        - Returns:
            - UnifiedMomentum_veer object
        - Example:
            >>> momentum = UnifiedMomentum_veer()
            >>> momentum_sol = momentum(Ctprime, yaw, veer=0.02, TI=0.05)

    __call__: 
        - Args: 
            - Ctprime (float): Thrust coefficient including the effect of yaw and tilt.
            - yaw (float, optional): Yaw angle of the rotor in radians
            - tilt (float, optional): Tilt angle of the rotor in radians
            - TI (float, optional): Turbulence intensity at the rotor location.
            - veer (float, optional): Veer angle at the rotor location.
    """

    def __init__(
        self, beta_s=0.1403, alpha=2.32, cached=True, v4_correction=1.0, **kwargs
    ):
        super().__init__(
            beta_s=beta_s, cached=cached, v4_correction=v4_correction, **kwargs
        )
        self.alpha = alpha

    def post_process(self, result, Ctprime, yaw=0, tilt=0, TI=0, veer=0):
        a, u4, v4, _x0, dp = result.x
        x0 = x0_model(u4, a, veer=veer, TI=TI, alpha=self.alpha, beta_s=self.beta_s)

        result.x = (a, u4, v4, x0, dp)  # correct x0
        return super().post_process(result, Ctprime, yaw=yaw, tilt=tilt)


def x0_model_scalar(u4, an, veer=0, TI=0, alpha=2.32, beta_s=0.1403):
    """
    Computes the near-wake length x0 considering veered
    conditions using a skewed ellipse approximation.

    Parameters
    ----------
    u4 : float or array-like
    an : float or array-like
    veer : float or array-like
        Amount of veering in radians per length
    TI : float or array-like
        Turbulence intensity TI = sqrt(2k/3) / U
    alpha : float
        TI dependence parameter, default is 2.32 (Bastankhah and Porté-Agel, 2016)
    beta_s : float
        Shear layer growth parameter, default is 0.1403 (Liew et al. (2024))

    Returns
    -------
    float or array-like
        Near-wake length x0
    """

    def _func(_x):
        c = _x * veer
        return (
            _x
            * (beta_s * np.abs((1 - u4) / (1 + u4)) + alpha * TI * np.abs(2 / (1 + u4)))
        ) - 0.5 * np.sqrt((1 - an) / (1 + u4)) * np.sqrt(
            0.5 * (c**2 + 2 - np.sqrt(c**4 + 4 * c**2))
        )

    return root(_func, x0=1).x[0]


def x0_model(u4, an, veer=0, TI=0, alpha=2.32, beta_s=0.1403):
    """
    Computes the near-wake length x0 considering veered
    conditions using a skewed ellipse approximation.

    Calls `x0_model_scalar` but vectorized.

    Parameters
    ----------
    u4 : float or array-like
    an : float or array-like
    veer : float or array-like
        Amount of veering in radians per length
    TI : float or array-like
        Turbulence intensity TI = sqrt(2k/3) / U
    alpha : float
        TI dependence parameter, default is 2.32 (Bastankhah and Porté-Agel, 2016)
    beta_s : float
        Shear layer growth parameter, default is 0.1403 (Liew et al. (2024))

    Returns
    -------
    float or array-like
        Near-wake length x0
    """

    f = np.vectorize(x0_model_scalar)
    return f(u4, an, veer=veer, TI=TI, alpha=alpha, beta_s=beta_s)
