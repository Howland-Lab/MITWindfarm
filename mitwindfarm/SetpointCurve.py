
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
    Thrust setpoint curve for an actuator disk operating above rated wind speed.

    """
    def __init__(self, rated: float = 10.59, cutout:float=25.0):
        self.AD_model = Heck()
        self.rated_ws = rated
        self.cutout = cutout
        self.windspeeds = np.linspace(2, cutout, 200)

        def get_optimal_power(windspeed):
            
            def f(x):
                return - self.AD_model(x, 0.0).Cp

            def constraint_func(x):
                cp = self.AD_model(x, 0.0).Cp[0]
                rotor_ws = windspeed
                return [(16/27) - (cp * ((rotor_ws / rated) ** 3))]

            constraint = dict(type="ineq", fun=constraint_func)
            sol = minimize(f, x0 = 0.0001, constraints=constraint)
            return sol.x[0]

        ws_sols = [get_optimal_power(windspeed) for windspeed in self.windspeeds]
        self.ctprimes = ws_sols


    def __call__(self, REWS):
        """
        Returns the 'greedy' individual control setpoint for the specified wind speed.
        """

        dimensional_REWS = REWS * self.rated_ws
        return np.interp(dimensional_REWS, self.windspeeds, self.ctprimes)
