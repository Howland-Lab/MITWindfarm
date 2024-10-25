from mitwindfarm import Windfield, BEM, UnifiedAD, AD, RotorSolution, FixedControlAD
from MITRotor.ReferenceTurbines import IEA15MW, NREL_5MW
import numpy as np

def test_bem_rotor():
    bem_rotor = BEM(NREL_5MW())
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

def test_bem_rotor():
    bem_rotor = BEM(NREL_5MW())
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





def test_fixed_control_rotor():
    rotor_model = FixedControlAD()
    Ctprime = rotor_model.setpoint_curve(1.0)
    print(Ctprime)
    assert np.isclose(Ctprime, 2.0, rtol = 5e-2)

if __name__ == "__main__":
    
    test_fixed_control_rotor()