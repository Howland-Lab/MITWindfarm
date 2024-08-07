
"""
Setpoint Curves

This module contains setpoint curves for turbines which operate with a fixed control strategy.

Classes:
- ReferenceTurbine: Abstract base class for reference turbines.
- IndividualControl: SetpointCurve for a turbine operating above rated wind speed at individual power optimum.
"""

import polars as pl
import numpy as np
import os

from scipy.optimize import minimize
from UnifiedMomentumModel.Momentum import Heck

class SetpointCurve():
    """
    Thrust setpoint curve for an actuator disk operating at individual control optimum.

    """
    def __init__(self):
        self.AD_model = Heck()
        self.windspeeds = np.linspace(0.01, 3.0, 2000)

        def get_optimal_power(ws):
            
            def f(x):
                return - self.AD_model(x, 0.0).Cp

            def constraint_func(x):
                Cp = self.AD_model(x, 0.0).Cp[0]
                Cp_max = ((16/27) * (1 / ws) ** 3)
                return Cp_max - Cp

            constraint = dict(type="ineq", fun=constraint_func)
            sol = minimize(f, x0 = 0.0001, constraints=constraint)
            return sol.x[0]

        ws_sols = [get_optimal_power(windspeed) for windspeed in self.windspeeds]
        self.ctprimes = ws_sols


    def __call__(self, REWS):
        """
        Returns the 'greedy' individual control setpoint for the specified wind speed.

        Parameters:
        - REWS (float): the wind speed at the turbine rotor normalized by the rated wind speed.

        Returns:
        - float: the Ct' control setpoint.
        """

        return np.interp(REWS, self.windspeeds, self.ctprimes)
