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

from UnifiedMomentumModel.Momentum import Heck, UnifiedMomentum, MomentumSolution
from MITRotor.BEM import BEM as _BEM
from MITRotor.BEM import BEMSolution, RotorDefinition


@dataclass
class RotorSolution:
    """
    Data class representing the solution of rotor models.

    Attributes:
    - Cp (float): Power coefficient.
    - Ct (float): Thrust coefficient.
    - Ctprime (float): Thrust coefficient including the effect of yaw.
    - an (float): Axial induction factor.
    - u4 (float): Axial velocity at the rotor tip.
    - v4 (float): Tangential velocity at the rotor tip.
    """

    Cp: float
    Ct: float
    Ctprime: float
    an: float
    u4: float
    v4: float


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


class BEM(Rotor):
    """
    Blade Element Momentum (BEM) rotor model.

    Attributes:
    - rotor_definition (RotorDefinition): Definition of the rotor parameters.

    Methods:
    - __call__(pitch, tsr, yaw): Calculate the rotor solution for given pitch, TSR, and yaw inputs.
    """

    def __init__(self, rotor_definition: RotorDefinition, **kwargs):
        """
        Initialize the BEM rotor model with the given rotor definition.

        Parameters:
        - rotor_definition (RotorDefinition): Definition of the rotor parameters.
        - **kwargs: Additional keyword arguments passed to the underlying BEM model.
        """
        self._model = _BEM(rotor_definition, **kwargs)

    def __call__(self, pitch, tsr, yaw) -> RotorSolution:
        """
        Calculate the rotor solution for given pitch, TSR, and yaw inputs.

        Parameters:
        - pitch (float): Pitch angle of the rotor blades.
        - tsr (float): Tip-speed ratio of the rotor.
        - yaw (float): Yaw angle of the rotor.

        Returns:
        RotorSolution: The calculated rotor solution.
        """
        sol: BEMSolution = self._model(pitch, tsr, yaw)
        return RotorSolution(
            sol.Cp(), sol.Ct(), sol.Ctprime(), sol.a(), sol.u4(), sol.v4()
        )


class UnifiedAD(Rotor):
    """
    Unified Momentum Model rotor with an axial induction factor.

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

    def __call__(self, Ctprime, yaw) -> RotorSolution:
        """
        Calculate the rotor solution for given Ctprime and yaw inputs.

        Parameters:
        - Ctprime (float): Thrust coefficient including the effect of yaw.
        - yaw (float): Yaw angle of the rotor.

        Returns:
        RotorSolution: The calculated rotor solution.
        """
        sol: MomentumSolution = self._model(Ctprime, yaw)
        return RotorSolution(
            sol.Cp[0], sol.Ct[0], sol.Ctprime, sol.an[0], sol.u4[0], sol.v4[0]
        )


class AD(Rotor):
    """
    Axial Distribution rotor model.

    Methods:
    - __call__(Ctprime, yaw): Calculate the rotor solution for given Ctprime and yaw inputs.
    """

    def __init__(self):
        """
        Initialize the AD rotor model using the Heck momentum model.
        """
        self._model = Heck()

    def __call__(self, Ctprime, yaw) -> RotorSolution:
        """
        Calculate the rotor solution for given Ctprime and yaw inputs.

        Parameters:
        - Ctprime (float): Thrust coefficient including the effect of yaw.
        - yaw (float): Yaw angle of the rotor.

        Returns:
        RotorSolution: The calculated rotor solution.
        """
        sol: MomentumSolution = self._model(Ctprime, yaw)
        return RotorSolution(sol.Cp, sol.Ct, sol.Ctprime, sol.an, sol.u4, sol.v4)
