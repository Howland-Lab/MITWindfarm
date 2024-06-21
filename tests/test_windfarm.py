import numpy as np

from mitwindfarm import Line, Layout, Windfarm, RefCtrlWindfarm, AnalyticalAvgWindfarm, AD, RefCtrlAD, RefCtrlAnalyticalAvgAD, RefCtrlAnalyticalAvgWindfarm

def test_ref_rotor():
    layout = Layout(xs = [0.0, 6.0, 12.0, 18.0], ys = [0.0, 0.0, 0.0, 0.0])
    windfarm = Windfarm()
    RCAA_windfarm = RefCtrlWindfarm(rotor_model=RefCtrlAD(u_rated=1.0))
    wf_sol = windfarm(layout = layout, setpoints= [(2.0, 0.0) for _ in layout.x])
    RCAA_sol = RCAA_windfarm(layout = layout, thrust_spts=[2.0, 2.0, 2.0, 2.0],
                            yaw_spts= [0.0, 0.0, 0.0, 0.0])

    assert np.array_equal([rot.Cp for rot in wf_sol.rotors],
                          [rot.Cp for rot in RCAA_sol.rotors])

def test_aa_ref_rotor():
    layout = Layout(xs = [0.0, 6.0, 12.0, 18.0], ys = [0.0, 0.0, 0.0, 0.0])
    windfarm = AnalyticalAvgWindfarm()
    RCAA_windfarm = RefCtrlAnalyticalAvgWindfarm(rotor_model=RefCtrlAnalyticalAvgAD(u_rated=1.0))
    wf_sol = windfarm(layout = layout, setpoints= [(2.0, 0.0) for _ in layout.x])
    RCAA_sol = RCAA_windfarm(layout = layout, thrust_spts=[2.0, 2.0, 2.0, 2.0],
                            yaw_spts= [0.0, 0.0, 0.0, 0.0])

    assert np.array_equal([rot.Cp for rot in wf_sol.rotors], [rot.Cp for rot in RCAA_sol.rotors])

def test_analytical_avg_wf():

    layout = Layout(xs = [0.0, 6.0, 0.0, 6.0], ys = [0.0, 0.0, 6.0, 6.0])
    windfarm = Windfarm(rotor_model=AD(rotor_grid=Line(disc=200)))
    wf_sol = windfarm(layout = layout, setpoints= [(2.0, 0.0) for _ in layout.x])
    aa_windfarm = AnalyticalAvgWindfarm()
    aa_wf_sol = aa_windfarm(layout = layout, setpoints= [(2.0, 0.0) for _ in layout.x])
    assert np.allclose([rot.Cp for rot in wf_sol.rotors], [rot.Cp for rot in aa_wf_sol.rotors], 0.01)



def test_ref_rotor2():
    layout = Layout(xs = [0.0, 6.0, 0.0, 6.0], ys = [0.0, 0.0, 6.0, 6.0])
    windfarm = Windfarm()
    RCAA_windfarm = RefCtrlWindfarm(rotor_model=RefCtrlAD(u_rated=1.0))
    wf_sol = windfarm(layout = layout, setpoints= [(2.0, 0.0) for _ in layout.x])
    RCAA_sol = RCAA_windfarm(layout = layout, thrust_spts=[2.0, 2.0, 2.0, 2.0],
                            yaw_spts= [0.0, 0.0, 0.0, 0.0])


    assert np.array_equal([rot.Cp for rot in wf_sol.rotors],
                          [rot.Cp for rot in RCAA_sol.rotors])

def test_aa_ref_rotor2():
    layout = Layout(xs = [0.0, 6.0, 0.0, 6.0], ys = [0.0, 0.0, 6.0, 6.0])
    windfarm = AnalyticalAvgWindfarm()
    RCAA_windfarm = RefCtrlAnalyticalAvgWindfarm(rotor_model=RefCtrlAnalyticalAvgAD(u_rated=1.0))
    wf_sol = windfarm(layout = layout, setpoints= [(2.0, 0.0) for _ in layout.x])
    RCAA_sol = RCAA_windfarm(layout = layout, thrust_spts=[2.0, 2.0, 2.0, 2.0],
                            yaw_spts= [0.0, 0.0, 0.0, 0.0])

    assert np.array_equal([rot.Cp for rot in wf_sol.rotors], [rot.Cp for rot in RCAA_sol.rotors])




