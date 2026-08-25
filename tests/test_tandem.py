"""
Smoke tests for the TANDEM closures migrated into `mitwindfarm.tandem`
(from the `kl_model` analysis repo's `mitwf_extras.py`).
"""

import numpy as np
from mitwindfarm import Uniform, Layout, ArbitraryXWindfield
from mitwindfarm.windfarm import CurledWindfarm

# importing this module registers "tandem", "kl-interp", "kl-hub", "scott", etc.
# into the CurledTurbulenceModel / CurledVModel / CurledWModel registries
import mitwindfarm.tandem as tandem


def get_curled_windfarm(k_model="tandem"):
    base_windfield = Uniform(TIamb=0.05)
    wf_curled = CurledWindfarm(
        base_windfield=base_windfield,
        solver_kwargs=dict(
            dy=1 / 5,
            dz=1 / 5,
            integrator="scipy_rk23",
            k_model=k_model,
            verbose=False,
        ),
    )
    layout = Layout([0], [0], [0])
    return wf_curled, layout


def test_kl_md_registered_and_runs():
    """CurledKL_MD should be registered under 'tandem' and solve without error."""
    assert tandem.TurbulenceModel_tandem.name in ["tandem"]
    wf_curled, layout = get_curled_windfarm(k_model="tandem")
    setpoints = [(1.33, 0.0, 0.0)]
    sol = wf_curled(layout, setpoints)
    assert np.isfinite(sol.rotors[0].Cp)


def test_scott_closure_registered_and_runs():
    """CurledTurbulenceModel_Scott should be registered under 'scott' and solve without error."""
    wf_curled, layout = get_curled_windfarm(k_model="scott")
    setpoints = [(1.33, 0.0, 0.0)]
    sol = wf_curled(layout, setpoints)
    assert np.isfinite(sol.rotors[0].Cp)


def test_arbitrary_x_windfield():
    """ArbitraryXWindfield should interpolate U along x with no y/z dependence."""
    x = np.array([0, 1, 2, 3])
    U_x = np.array([8.0, 7.0, 6.0, 5.0])
    field = ArbitraryXWindfield(x=x, U_x=U_x)
    wsp = field.wsp(1.5, 0.0, 0.0)
    assert np.isclose(wsp, 6.5)


if __name__ == "__main__":
    test_kl_md_registered_and_runs()
    test_scott_closure_registered_and_runs()
    test_arbitrary_x_windfield()
    print("All TANDEM tests passed.")
