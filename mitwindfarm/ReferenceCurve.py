"""
Reference Turbines

This module contains reference turbine models to be used in wind farm layout
and control optimizations.

Classes:
- ReferenceTurbine: Abstract base class for reference turbines.
- IEA15: IEA 15 MW Reference Turbine."""

from abc import ABC, abstractmethod
import polars as pl
import numpy as np
import os

path = os.path.dirname(__file__)


class ReferenceCurve(ABC):
    """
    Thrust and power curves of reference turbine.

    """

    # @abstractmethod
    # def __call__(self, REWS: float) -> float:
    #     """
    #     Parameters:
    #         REWS (float): Rotor equivalent wind speed normalized by the rated
    #             wind speed of the turbine.

    #     Returns:
    #         Ctprime (float): The Ctprime setpoint for the turbine based on the
    #             reference model."""
    #     pass


class ReferenceCurve_IEA15MW(ReferenceCurve):
    def __init__(self):
        self.cutin_ws = 3
        self.rated_ws = 10.59
        self.windspeeds = np.array(
            [
                0.0,
                0.6,
                1.2,
                1.8,
                2.4,
                3.0,
                3.54953237,
                4.06790077,
                4.55390685,
                5.00642706,
                5.42441529,
                5.80690523,
                6.15301265,
                6.46193743,
                6.7329654,
                6.96547,
                7.15891374,
                7.31284942,
                7.42692116,
                7.50086527,
                7.5345108,
                7.54124163,
                7.58833327,
                7.67567684,
                7.80307043,
                7.97021953,
                8.17673773,
                8.4221476,
                8.70588182,
                9.02728444,
                9.38561247,
                9.78003751,
                10.20964776,
                10.65845809,
                10.67345004,
                11.17037214,
                11.6992653,
                12.25890683,
                12.84800295,
                13.46519181,
                14.10904661,
                14.77807889,
                15.470742,
                16.18543466,
                16.92050464,
                17.67425264,
                18.44493615,
                19.23077353,
                20.02994808,
                20.8406123,
                21.66089211,
                22.4888912,
                23.32269542,
                24.1603772,
                25.0,
            ]
        )
        self.ctprimes = np.array(
            [
                1.00000000e-05,
                1.00000000e-05,
                1.00000000e-05,
                1.00000000e-05,
                1.00000000e-05,
                1.56004449e00,
                1.46428500e00,
                1.45274999e00,
                1.46719154e00,
                1.47800724e00,
                1.48567967e00,
                1.48790816e00,
                1.48545900e00,
                1.48078924e00,
                1.47521530e00,
                1.46928376e00,
                1.43893613e00,
                1.43893613e00,
                1.43893613e00,
                1.43893613e00,
                1.43893613e00,
                1.43893613e00,
                1.43893613e00,
                1.43893613e00,
                1.43893613e00,
                1.43893613e00,
                1.43893613e00,
                1.43893613e00,
                1.43893613e00,
                1.43893613e00,
                1.43893613e00,
                1.43893613e00,
                1.43893613e00,
                1.41366160e00,
                1.32324664e00,
                8.14752600e-01,
                6.17666525e-01,
                4.90317183e-01,
                3.98882371e-01,
                3.29790635e-01,
                2.75965920e-01,
                2.33157309e-01,
                1.98601658e-01,
                1.70385717e-01,
                1.47132275e-01,
                1.27819815e-01,
                1.11673140e-01,
                9.80935494e-02,
                8.66120609e-02,
                7.68569589e-02,
                6.85303755e-02,
                6.13942136e-02,
                5.52522158e-02,
                4.99459840e-02,
                4.53450629e-02,
            ]
        )

        self.cps = np.array(
            [
                0.05897929,
                0.244877465,
                0.338251306,
                0.389145491,
                0.418518725,
                0.436197187,
                0.447160954,
                0.454108003,
                0.458559593,
                0.461416015,
                0.463239112,
                0.46363055,
                0.46363055,
                0.46363055,
                0.46363055,
                0.46363055,
                0.46363055,
                0.46363055,
                0.46363055,
                0.46363055,
                0.46363055,
                0.46363055,
                0.46363055,
                0.46363055,
                0.46363055,
                0.46363055,
                0.46363055,
                0.46363055,
                0.46383339,
                0.46187796,
                0.402937693,
                0.350724509,
                0.304850149,
                0.264810223,
                0.230039513,
                0.19996361,
                0.174015997,
                0.151673324,
                0.13245527,
                0.115931703,
                0.101722923,
                0.089497401,
                0.078968073,
                0.069887942,
                0.062045494,
                0.055260221,
                0.049379055,
                0.044269582,
                0.039822686,
                0.035943611,
            ]
        )

    def thrust(self, REWS):
        dimensional_REWS = REWS * self.rated_ws
        return np.interp(dimensional_REWS, self.windspeeds, self.ctprimes)

    def power(self, REWS):
        dimensional_REWS = REWS * self.rated_ws
        return np.interp(dimensional_REWS, self.windspeeds, self.cps)
