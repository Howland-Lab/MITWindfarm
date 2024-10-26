import numpy as np
import matplotlib.pyplot as plt
from mitwindfarm import Windfarm, AnalyticalWindfarm, Windfarm, Line, AD, Layout


windfarm_lineavg = AnalyticalWindfarm(TIamb= 0.07)
windfarm_lineavg.wake_model.WATI_sigma_multiplier = 2.0
windfarm_sample = Windfarm(rotor_model=AD(rotor_grid=Line()), TIamb= 0.07)
windfarm_sample.wake_model.WATI_sigma_multiplier = 2.0

def get_rews_sample(y):
    layout = Layout([0.0, 10.0], [0.0, y])
    setpoints = [(2.0, 0.0), (2.0, 0.0)]
    sol = windfarm_sample(layout=layout, setpoints=setpoints)
    return sol.rotors[1].REWS

def get_rews_lineavg(y):
    layout = Layout([0.0, 10.0], [0.0, y])
    setpoints = [(2.0, 0.0), (2.0, 0.0)]
    sol = windfarm_lineavg(layout=layout, setpoints=setpoints)
    return sol.rotors[1].REWS

def get_RETI_sample(y):
    layout = Layout([0.0, 10.0], [0.0, y])
    setpoints = [(2.0, 0.0), (2.0, 0.0)]
    sol = windfarm_sample(layout=layout, setpoints=setpoints)
    return sol.rotors[1].TI

def get_RETI_lineavg(y):
    layout = Layout([0.0, 10.0], [0.0, y])
    setpoints = [(2.0, 0.0), (2.0, 0.0)]
    sol = windfarm_lineavg(layout=layout, setpoints=setpoints)
    return sol.rotors[1].TI

ys = np.linspace(-5, 5, 150)
line_rews = [get_rews_lineavg(y) for y in ys]
sample_rews = [get_rews_sample(y) for y in ys]

# plt.plot(ys, line_rews, label = "line")
# plt.plot(ys, sample_rews, label = "sample")
# plt.legend()
# plt.show()

line_ti = [get_RETI_lineavg(y) for y in ys]
sample_ti = [get_RETI_sample(y) for y in ys]

plt.plot(ys, line_ti, label = "line")
plt.plot(ys, sample_ti, label = "sample")
plt.legend()
plt.show()