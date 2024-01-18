from dataclasses import dataclass
from typing import Optional

import numpy as np

from .Layout import Layout
from .Rotor import Rotor, AD
from .Windfield import Windfield, Uniform
from .Wake import WakeModel, Wake, GaussianWakeModel
from .Superposition import Superposition, Linear


@dataclass
class WindfarmSolution:
    layout: Layout
    setpoints: list[tuple]
    rotors: list[Rotor]
    wakes: list[Wake]
    windfield: Windfield

    def Cp(self):
        return np.mean([x.Cp for x in self.rotors])


class Windfarm:
    def __init__(
        self,
        rotor_model: Optional[Rotor] = None,
        wake_model: Optional[WakeModel] = None,
        superposition: Optional[Superposition] = None,
        base_windfield: Optional[Windfield] = None,
    ):
        self.rotor_model = AD() if rotor_model is None else rotor_model
        self.wake_model = GaussianWakeModel() if wake_model is None else wake_model
        self.superposition = Linear() if superposition is None else superposition
        self.base_windfield = Uniform() if base_windfield is None else base_windfield

    def __call__(
        self, layout: Layout, setpoints: list[tuple[float, ...]]
    ) -> WindfarmSolution:
        N = len(layout)
        wakes = N * [None]
        rotor_solutions = N * [None]

        windfield = self.superposition(self.base_windfield, [])
        for i, (x, y, z) in layout.iter_downstream():
            rotor_solutions[i] = self.rotor_model(x, y, z, windfield, *setpoints[i])
            wakes[i] = self.wake_model(x, y, z, rotor_solutions[i])
            windfield.add_wake(wakes[i])

        return WindfarmSolution(layout, setpoints, rotor_solutions, wakes, windfield)
