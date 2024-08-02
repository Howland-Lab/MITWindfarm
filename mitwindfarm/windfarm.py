from dataclasses import dataclass, asdict
from typing import Optional, Union

import numpy as np

from ._Layout import Layout
from .Rotor import Rotor, AD, AnalyticalAD, AnalyticalUnifiedAD, RotorSolution
from .Windfield import Windfield, Uniform
from .Wake import WakeModel, Wake, GaussianWakeModel, VariableKwGaussianWakeModel
from .Superposition import Superposition, Linear


@dataclass
class WindfarmSolution:
    layout: Layout
    setpoints: list[tuple]
    rotors: list[RotorSolution]
    wakes: list[Wake]
    windfield: Windfield

    @property
    def Cp(self):
        return np.mean([x.Cp for x in self.rotors])

    def to_dict(self) -> dict:
        return PartialWindfarmSolution.from_WindfarmSolution(self).to_dict()


@dataclass
class PartialWindfarmSolution:
    layout: Layout
    setpoints: list[tuple]
    rotors: list[RotorSolution]

    @classmethod
    def from_WindfarmSolution(cls, sol: WindfarmSolution) -> "PartialWindfarmSolution":
        return PartialWindfarmSolution(sol.layout, sol.setpoints, sol.rotors)

    def to_dict(self) -> dict:
        out = dict(
            layout=[(x, y, z) for x, y, z in self.layout],
            setpoints=self.setpoints,
            rotors=[asdict(rotor) for rotor in self.rotors],
        )

        return out

    @classmethod
    def from_dict(cls, input: dict) -> "PartialWindfarmSolution":
        xs = [loc[0] for loc in input["layout"]]
        ys = [loc[1] for loc in input["layout"]]
        zs = [loc[2] for loc in input["layout"]]

        layout = Layout(xs, ys, zs)

        rotors = [RotorSolution(**sol) for sol in input["rotors"]]

        return cls(layout, input["setpoints"], rotors)


class Windfarm:
    def __init__(
        self,
        rotor_model: Optional[Rotor] = None,
        wake_model: Optional[WakeModel] = None,
        superposition: Optional[Superposition] = None,
        base_windfield: Optional[Windfield] = None,
        TIamb: float = None,
    ):
        self.rotor_model = AD() if rotor_model is None else rotor_model
        self.wake_model = GaussianWakeModel() if wake_model is None else wake_model
        self.superposition = Linear() if superposition is None else superposition
        self.base_windfield = Uniform(TIamb=TIamb) if base_windfield is None else base_windfield
        self.TIamb = TIamb

    def __call__(self, layout: Layout, setpoints: list[tuple[float, ...]]) -> WindfarmSolution:
        N = layout.x.size
        wakes = N * [None]
        rotor_solutions = N * [None]

        windfield = self.superposition(self.base_windfield, [])
        for i, (x, y, z) in layout.iter_downstream():
            rotor_solutions[i] = self.rotor_model(x, y, z, windfield, *setpoints[i])
            rotor_solutions[i].idx = i
            wakes[i] = self.wake_model(x, y, z, rotor_solutions[i], TIamb=self.TIamb)
            windfield.add_wake(wakes[i])

        return WindfarmSolution(layout, setpoints, rotor_solutions, wakes, windfield)

    def from_partial(self, partial: PartialWindfarmSolution) -> WindfarmSolution:
        N = len(partial.layout)
        wakes = N * [None]
        windfield = self.superposition(self.base_windfield, [])
        for i, (x, y, z) in partial.layout.iter_downstream():
            wakes[i] = self.wake_model(x, y, z, partial.rotors[i])
            windfield.add_wake(wakes[i])

        return WindfarmSolution(partial.layout, partial.setpoints, partial.rotors, wakes, windfield)

    def from_dict(self, partial: dict) -> WindfarmSolution:
        return self.from_partial(PartialWindfarmSolution.from_dict(partial))


class AnalyticalWindfarm(Windfarm):
    def __init__(
        self,
        rotor_model: Union[AnalyticalAD, AnalyticalUnifiedAD] = None,
        wake_model: Union[GaussianWakeModel, VariableKwGaussianWakeModel] = None,
        base_windfield: Optional[Windfield] = None,
        TIamb: float = None
    ):
        self.rotor_model = AnalyticalAD() if rotor_model is None else rotor_model
        self.wake_model = GaussianWakeModel() if wake_model is None else wake_model
        self.superposition = Linear()
        self.base_windfield = Uniform(TIamb=TIamb) if base_windfield is None else base_windfield
        self.TIamb = TIamb
