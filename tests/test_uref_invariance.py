"""
Test that the CurledWake solver produces non-dimensionally invariant results
when the freestream velocity U0 is rescaled (and k-l is used).
"""

import numpy as np
from pytest import approx
from mitwindfarm import Uniform, Layout
from mitwindfarm.windfarm import CurledWindfarm
from mitwindfarm.Rotor import UnifiedAD_TI


def _run_curled_farm(U0, k_model="const"):
    """Run a 2-turbine yawed farm at given U0."""
    wf = CurledWindfarm(
        rotor_model=UnifiedAD_TI(),
        base_windfield=Uniform(U0=U0, TIamb=0.05),
        solver_kwargs=dict(
            dy=0.1, dz=0.1, dx=0.05,
            integrator="ef",
            k_model=k_model,
            verbose=False,
        ),
    )
    layout = Layout([0, 5], [0, 0], [0, 0])
    yaw = np.radians(20)
    setpoints = [(2, yaw, 0)] * 2
    return wf(layout, setpoints)


def test_uref_invariance_kl():
    """
    REWS/U0 must be identical at U0=1 and U0=2 (k-l turbulence model).
    
    Note: "const" will not be u-invariant unless the eddy viscosity `nu_T` is 
    also scaled with U0.
    """
    sol1 = _run_curled_farm(U0=1.0, k_model="k-l")
    sol2 = _run_curled_farm(U0=2.0, k_model="k-l")

    for i in range(2):
        rews_ratio_1 = sol1.rotors[i].REWS / 1.0
        rews_ratio_2 = sol2.rotors[i].REWS / 2.0
        assert rews_ratio_2 == approx(rews_ratio_1, rel=1e-5), (
            f"Turbine {i}: REWS/U0 differs between U0=1 ({rews_ratio_1:.4f}) "
            f"and U0=2 ({rews_ratio_2:.4f})"
        )


def test_downstream_rews_below_freestream():
    """Downstream turbine in a wake cannot see REWS > U0 with uniform inflow."""
    sol = _run_curled_farm(U0=8.0, k_model="const")
    for i in range(1, 2):  # skip turbine 0 (upstream)
        assert sol.rotors[i].REWS <= 8.0 * 1.01, (
            f"Turbine {i}: REWS={sol.rotors[i].REWS:.2f} exceeds U0=8.0 "
            f"— physically impossible with uniform inflow"
        )
