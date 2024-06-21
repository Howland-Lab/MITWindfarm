import numpy as np

from mitwindfarm import Line, Layout, Windfarm, ReferenceWindfarm, AnalyticalAvgWindfarm, AD, ReferenceRotor, AnalyticalAvgReferenceRotor, AnalyticalAvgReferenceWindfarm


def test_analytical_avg_wf():

    layout = Layout(xs = [0.0, 6.0, 0.0, 6.0], ys = [0.0, 0.0, 6.0, 6.0])
    windfarm = Windfarm(rotor_model=AD(rotor_grid=Line(disc=200)))
    wf_sol = windfarm(layout = layout, setpoints= [(2.0, 0.0) for _ in layout.x])
    aa_windfarm = AnalyticalAvgWindfarm()
    aa_wf_sol = aa_windfarm(layout = layout, setpoints= [(2.0, 0.0) for _ in layout.x])
    assert np.allclose([rot.Cp for rot in wf_sol.rotors], [rot.Cp for rot in aa_wf_sol.rotors], 0.01)



