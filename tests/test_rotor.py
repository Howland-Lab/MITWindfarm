import numpy as np

from mitwindfarm import Windfield, BEM, UnifiedAD, AD, RotorSolution, Layout, Windfarm, RefCtrlWindfarm, AnalyticalAvgWindfarm, RefCtrlAD, RefCtrlAnalyticalAvgAD, RefCtrlAnalyticalAvgWindfarm
from MITRotor.ReferenceTurbines import IEA15MW


def test_bem_rotor():
    bem_rotor = BEM(IEA15MW())
    x, y, z = 0, 0, 0
    windfield = Windfield.Uniform()

    pitch = 0.0
    tsr = 5.0
    yaw = 0.0

    solution = bem_rotor(x, y, z, windfield, pitch, tsr, yaw)

    assert isinstance(solution, RotorSolution)
    assert isinstance(solution.Cp, float)
    assert isinstance(solution.Ct, float)
    assert isinstance(solution.Ctprime, float)
    assert isinstance(solution.an, float)
    assert isinstance(solution.u4, float)
    assert isinstance(solution.v4, float)


def test_unified_ad_rotor():
    unified_ad_rotor = UnifiedAD(beta=0.1403)
    x, y, z = 0, 0, 0
    windfield = Windfield.Uniform()

    Ctprime = 0.8
    yaw = 0.0

    solution = unified_ad_rotor(x, y, z, windfield, Ctprime, yaw)

    assert isinstance(solution, RotorSolution)
    assert isinstance(solution.Cp, float)
    assert isinstance(solution.Ct, float)
    assert isinstance(solution.Ctprime, float)
    assert isinstance(solution.an, float)
    assert isinstance(solution.u4, float)
    assert isinstance(solution.v4, float)


def test_ad_rotor():
    ad_rotor = AD()
    x, y, z = 0, 0, 0
    windfield = Windfield.Uniform()

    Ctprime = 0.7
    yaw = 0.0

    solution = ad_rotor(x, y, z, windfield, Ctprime, yaw)

    assert isinstance(solution, RotorSolution)
    assert isinstance(solution.Cp, float)
    assert isinstance(solution.Ct, float)
    assert isinstance(solution.Ctprime, float)
    assert isinstance(solution.an, float)
    assert isinstance(solution.u4, float)
    assert isinstance(solution.v4, float)


    
