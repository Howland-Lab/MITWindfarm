from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from UnifiedMomentumModel.Momentum import MomentumBase, Heck, MomentumSolution
from .Layout import Layout
from .Rotor import Rotor, AD, RotorSolution
from .RotorGrid import RotorGrid, Area
from .Windfield import Windfield, Uniform
from .Wake import WakeModel, Wake, GaussianWakeModel
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
    ):
        self.rotor_model = AD() if rotor_model is None else rotor_model
        self.wake_model = GaussianWakeModel() if wake_model is None else wake_model
        self.superposition = Linear() if superposition is None else superposition
        self.base_windfield = Uniform() if base_windfield is None else base_windfield

    def __call__(
        self, layout: Layout, setpoints: list[tuple[float, ...]]
    ) -> WindfarmSolution:
        N = layout.x.size
        wakes = N * [None]
        rotor_solutions = N * [None]

        windfield = self.superposition(self.base_windfield, [])
        for i, (x, y, z) in layout.iter_downstream():
            rotor_solutions[i] = self.rotor_model(x, y, z, windfield, *setpoints[i])
            wakes[i] = self.wake_model(x, y, z, rotor_solutions[i])
            windfield.add_wake(wakes[i])

        return WindfarmSolution(layout, setpoints, rotor_solutions, wakes, windfield)

    def from_partial(self, partial: PartialWindfarmSolution) -> WindfarmSolution:
        N = len(partial.layout)
        wakes = N * [None]
        windfield = self.superposition(self.base_windfield, [])
        for i, (x, y, z) in partial.layout.iter_downstream():
            wakes[i] = self.wake_model(x, y, z, partial.rotors[i])
            windfield.add_wake(wakes[i])

        return WindfarmSolution(
            partial.layout, partial.setpoints, partial.rotors, wakes, windfield
        )

    def from_dict(self, partial: dict) -> WindfarmSolution:
        return self.from_partial(PartialWindfarmSolution.from_dict(partial))
    
class RefTurbine:
    def __init__(
        self,
        ws_lookup,
        ctp_lookup,
    ):
        self.ws_lookup = ws_lookup
        self.ctp_lookup = ctp_lookup

    
class RefWindfarm:
    """
    Wind farm model which takes power/thrust curve based turbine models.
    """
    def __init__(
        self,
        reference_turbine: RefTurbine,
        dim_freestream_vel: float,
        rotor_model: Optional[Rotor] = None,
        wake_model: Optional[WakeModel] = None,
        superposition: Optional[Superposition] = None,
        base_windfield: Optional[Windfield] = None
    ):
        self.rotor_model = AD() if rotor_model is None else rotor_model
        self.wake_model = GaussianWakeModel() if wake_model is None else wake_model
        self.superposition = Linear() if superposition is None else superposition
        self.base_windfield = Uniform() if base_windfield is None else base_windfield
        self.reference_turbine = reference_turbine
        self.dim_frestream_vel = dim_freestream_vel

    def __call__(
        self, layout: Layout
    ) -> WindfarmSolution:
        N = layout.x.size
        wakes = N * [None]
        rotor_solutions = N * [None]
        setpoints = N * [None]

        windfield = self.superposition(self.base_windfield, [])
        for i, (x, y, z) in layout.iter_downstream():
            ws_at_rotor = windfield.wsp(x,y,z)
            ctp = np.interp(self.dim_frestream_vel * ws_at_rotor,
                            self.reference_turbine.ws_lookup,
                            self.reference_turbine.ctp_lookup)
            setpoints[i] = (ctp, 0.0)
            rotor_solutions[i] = self.rotor_model(x, y, z, windfield, *setpoints[i])
            wakes[i] = self.wake_model(x, y, z, rotor_solutions[i])
            windfield.add_wake(wakes[i])


        return WindfarmSolution(layout, setpoints, rotor_solutions, wakes, windfield)

    def from_partial(self, partial: PartialWindfarmSolution) -> WindfarmSolution:
        N = len(partial.layout)
        wakes = N * [None]
        windfield = self.superposition(self.base_windfield, [])
        for i, (x, y, z) in partial.layout.iter_downstream():
            wakes[i] = self.wake_model(x, y, z, partial.rotors[i])
            windfield.add_wake(wakes[i])

        return WindfarmSolution(
            partial.layout, partial.setpoints, partial.rotors, wakes, windfield
        )

    def from_dict(self, partial: dict) -> WindfarmSolution:
        return self.from_partial(PartialWindfarmSolution.from_dict(partial))
    

