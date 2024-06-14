"""
Unified Momentum Model Rotor Definitions

This module defines classes representing different rotor models based on the Unified Momentum Model.
It includes abstract classes and concrete implementations such as BEM, UnifiedAD, and AD.

Classes:
- Rotor: Abstract base class for rotor models.
- BEM: Blade Element Momentum (BEM) rotor model.
- UnifiedAD: Unified Momentum Model with an axial induction factor.
- AD: Axial Distribution rotor model.

Data Classes:
- RotorSolution: Data class representing the solution of rotor models.

Usage Example:
    rotor_def = RotorDefinition(...)  # Define rotor parameters
    bem_rotor = BEM(rotor_def)         # Create a BEM rotor instance
    solution = bem_rotor(pitch, tsr, yaw)  # Calculate rotor solution for given inputs
    print(solution.Cp, solution.Ct, solution.Ctprime, solution.an, solution.u4, solution.v4)

Note: Make sure to replace '...' with the actual parameters in RotorDefinition.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from UnifiedMomentumModel.Momentum import Heck, UnifiedMomentum, MomentumSolution
from MITRotor import BEM as _BEM
from MITRotor import BEMSolution, RotorDefinition
from .Windfield import Windfield
from .RotorGrid import RotorGrid, Point, Line, Area
from dualitic import DualNumber


@dataclass
class RotorSolution:
    """
    Data class representing the solution of rotor models.
    """

    yaw: float
    Cp: float
    Ct: float
    Ctprime: float
    an: float
    u4: float
    v4: float
    REWS: float
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

    Methods:
    - __call__(Ctprime, yaw): Calculate the rotor solution for given Ctprime and yaw inputs.
    """

    def __init__(self, rotor_grid: RotorGrid = None):
        """
        Initialize the AD rotor model using the Heck momentum model.
        """
        self._model = Heck()
        if rotor_grid is None:
            self.rotor_grid = Area()
        else:
            self.rotor_grid = rotor_grid

    def __call__(self, x: float, y: float, z: float, windfield: Windfield, Ctprime, yaw) -> RotorSolution:
        """
        Calculate the rotor solution for given Ctprime and yaw inputs.

        Parameters:
        - Ctprime (float): Thrust coefficient including the effect of yaw.
        - yaw (float): Yaw angle of the rotor.

        Returns:
        RotorSolution: The calculated rotor solution.
        """
        # Calculate rotor solution (independent of wind field in this model)
        sol: MomentumSolution = self._model(Ctprime, yaw)

        # Get the points over rotor to be sampled in windfield
        xs_loc, ys_loc, zs_loc = self.rotor_grid.grid_points()
        xs_glob, ys_glob, zs_glob = xs_loc + x, ys_loc + y, zs_loc + z

        # sample windfield and calculate rotor effective wind speed
        Us = windfield.wsp(xs_glob, ys_glob, zs_glob)
        REWS = self.rotor_grid.average(Us)
        
        # compute rotor equivalent turbulence intensity
        x_call = x if isinstance(x, DualNumber) else np.array([x])
        y_call = y if isinstance(y, DualNumber) else np.array([y])
        z_call = z if isinstance(z, DualNumber) else np.array([z])
        RETI = np.mean(windfield.RETI(x_call, y_call, z_call))

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

class AnalyticalAvgAD(Rotor):
    """
    Axial Distribution rotor model using analytically line averaged REWS.
    Methods:
    - __call__(Ctprime, yaw): Calculate the rotor solution for given Ctprime and yaw inputs.
    """

    def __init__(self):
        """
        Initialize the AD rotor model using the Heck momentum model.
        """
        self._model = Heck()

    def __call__(self, x: float, y: float, z: float, windfield: Windfield, Ctprime, yaw) -> RotorSolution:
        """
        Calculate the rotor solution for given Ctprime and yaw inputs.
        Parameters:
        - Ctprime (float): Thrust coefficient including the effect of yaw.
        - yaw (float): Yaw angle of the rotor.
        Returns:
        RotorSolution: The calculated rotor solution.
        """
        # Calculate rotor solution (independent of wind field in this model)
        sol: MomentumSolution = self._model(Ctprime, yaw)

        # sample analytically line-averaged rotor effective wind speed
        REWS = windfield.RE_wsp(x, y , z)

        # compute rotor equivalent turbulence intensity
        x_call = x if isinstance(x, DualNumber) else np.array([x])
        y_call = y if isinstance(y, DualNumber) else np.array([y])
        z_call = z if isinstance(z, DualNumber) else np.array([z])
        RETI = np.mean(windfield.RETI(x_call, y_call, z_call))

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


class UnifiedAD(Rotor):
    """
    Unified Momentum Model rotor with an axial induction factor.

    Attributes:
    - beta (float): Axial induction factor.

    Methods:
    - __call__(Ctprime, yaw): Calculate the rotor solution for given Ctprime and yaw inputs.
    """

    def __init__(self, rotor_grid: RotorGrid = None, beta=0.1403):
        """
        Initialize the UnifiedAD rotor model with the given axial induction factor.

        Parameters:
        - beta (float): Axial induction factor (default is 0.1403).
        """
        if rotor_grid is None:
            self.rotor_grid = Point()
        else:
            self.rotor_grid = rotor_grid
        self._model = UnifiedMomentum(beta=beta)

    def __call__(self, x: float, y: float, z: float, windfield: Windfield, Ctprime, yaw) -> RotorSolution:
        """
        Calculate the rotor solution for given Ctprime and yaw inputs.

        Parameters:
        - Ctprime (float): Thrust coefficient including the effect of yaw.
        - yaw (float): Yaw angle of the rotor.

        Returns:
        RotorSolution: The calculated rotor solution.
        """
        sol: MomentumSolution = self._model(Ctprime, yaw)

        # Get the points over rotor to be sampled in windfield
        xs_loc, ys_loc, zs_loc = self.rotor_grid.grid_points()
        xs_glob, ys_glob, zs_glob = xs_loc + x, ys_loc + y, zs_loc + z

        # sample windfield and calculate rotor effective wind speed
        Us = windfield.wsp(xs_glob, ys_glob, zs_glob)
        REWS = self.rotor_grid.average(Us)

        # compute rotor equivalent turbulence intensity
        x_call = x if isinstance(x, DualNumber) else np.array([x])
        y_call = y if isinstance(y, DualNumber) else np.array([y])
        z_call = z if isinstance(z, DualNumber) else np.array([z])
        RETI = np.mean(windfield.RETI(x_call, y_call, z_call))

        # rotor solution is normalised by REWS. Convert normalisation to U_inf and return
        return RotorSolution(
            yaw,
            sol.Cp[0] * REWS**3,
            sol.Ct[0] * REWS**2,
            sol.Ctprime,
            sol.an[0] * REWS,
            sol.u4[0] * REWS,
            sol.v4[0] * REWS,
            REWS,
            TI=RETI,
            extra=sol,
        )

class AnalyticalAvgUnifiedAD(Rotor):
    """
    Unified Momentum Model rotor with an axial induction factor using
        analytically line averaged REWS.
    Attributes:
    - beta (float): Axial induction factor.
    Methods:
    - __call__(Ctprime, yaw): Calculate the rotor solution for given Ctprime and yaw inputs.
    """

    def __init__(self, beta=0.1403):
        """
        Initialize the UnifiedAD rotor model with the given axial induction factor.
        Parameters:
        - beta (float): Axial induction factor (default is 0.1403).
        """
        self._model = UnifiedMomentum(beta=beta)

    def __call__(self, x: float, y: float, z: float, windfield: Windfield, Ctprime, yaw) -> RotorSolution:
        """
        Calculate the rotor solution for given Ctprime and yaw inputs.
        Parameters:
        - Ctprime (float): Thrust coefficient including the effect of yaw.
        - yaw (float): Yaw angle of the rotor.
        Returns:
        RotorSolution: The calculated rotor solution.
        """
        sol: MomentumSolution = self._model(Ctprime, yaw)

        # sample analytically line-averaged rotor effective wind speed
        REWS = windfield.RE_wsp(x, y , z)

        # compute rotor equivalent turbulence intensity
        x_call = x if isinstance(x, DualNumber) else np.array([x])
        y_call = y if isinstance(y, DualNumber) else np.array([y])
        z_call = z if isinstance(z, DualNumber) else np.array([z])
        RETI = np.mean(windfield.RETI(x_call, y_call, z_call))

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


class BEM(Rotor):
    """
    Blade Element Momentum (BEM) rotor model. Note: MITRotor is formulated in
    terms of rotor radius, whereas MITWindfarm is in rotor diameters.
    Conversions MUST be made between the two normalizations in this class.

    Attributes: - rotor_definition (RotorDefinition): Definition of the rotor
    parameters.

    Methods: - __call__(pitch, tsr, yaw): Calculate the rotor solution for given
    pitch, TSR, and yaw inputs.
    """

    def __init__(self, rotor_definition: RotorDefinition, BEM_model=None, **kwargs):
        """
        Initialize the BEM rotor model with the given rotor definition.

        Parameters:
        - rotor_definition (RotorDefinition): Definition of the rotor parameters.
        - **kwargs: Additional keyword arguments passed to the underlying BEM model.
        """
        BEM_model = BEM_model or _BEM
        self._model = BEM_model(rotor_definition, **kwargs)
        self.xgrid_loc, self.ygrid_loc, self.zgrid_loc = self._model.sample_points()
        # Convert from radius to diameter normalization
        self.xgrid_loc /= 2
        self.ygrid_loc /= 2
        self.zgrid_loc /= 2

    def __call__(self, x: float, y: float, z: float, windfield: Windfield, pitch, tsr, yaw) -> RotorSolution:
        """
        Calculate the rotor solution for given pitch, TSR, and yaw inputs.

        Parameters:
        - pitch (float): Pitch angle of the rotor blades.
        - tsr (float): Tip-speed ratio of the rotor.
        - yaw (float): Yaw angle of the rotor.

        Returns:
        RotorSolution: The calculated rotor solution.
        """
        xs_glob = self.xgrid_loc + x
        ys_glob = self.ygrid_loc + y
        zs_glob = self.zgrid_loc + z
        Us, TIs = windfield.wsp_and_TI(xs_glob, ys_glob, zs_glob)
        REWS = self._model.geometry.rotor_average(self._model.geometry.annulus_average(Us))
        RETI = np.sqrt(self._model.geometry.rotor_average(self._model.geometry.annulus_average(TIs**2)))

        wdir = windfield.wdir(xs_glob, ys_glob, zs_glob)
        sol: BEMSolution = self._model(pitch, tsr, yaw, Us / REWS, wdir)
        return RotorSolution(
            yaw,
            sol.Cp() * REWS**3,
            sol.Ct() * REWS**2,
            sol.Ctprime(),
            sol.a() * REWS,
            sol.u4() * REWS,
            sol.v4() * REWS,
            REWS,
            TI=RETI,
            extra=sol,
        )
