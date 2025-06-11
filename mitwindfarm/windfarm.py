from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from ._Layout import Layout
from .Rotor import Rotor, AD, RotorSolution
from .Windfield import Windfield, Uniform
from .Wake import WakeModel, Wake, GaussianWakeModel
from .CurledWake import CurledWakeWindfield
from .Superposition import Superposition, Niayifar


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
        self.superposition = Niayifar() if superposition is None else superposition
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


class CosineWindfarm:
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
        self.superposition = Niayifar() if superposition is None else superposition
        self.base_windfield = Uniform(TIamb=TIamb) if base_windfield is None else base_windfield
        self.TIamb = TIamb

    def __call__(self, layout: Layout, yaw_setpoints: list[float]) -> WindfarmSolution:
        N = layout.x.size
        wakes = N * [None]
        rotor_solutions = N * [None]

        windfield = self.superposition(self.base_windfield, [])
        for i, (x, y, z) in layout.iter_downstream():
            rotor_solutions[i] = self.rotor_model(x, y, z, windfield, yaw_setpoints[i])
            rotor_solutions[i].idx = i
            wakes[i] = self.wake_model(x, y, z, rotor_solutions[i], TIamb=self.TIamb)
            windfield.add_wake(wakes[i])

        return WindfarmSolution(layout, yaw_setpoints, rotor_solutions, wakes, windfield)

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


class CurledWindfarm(Windfarm):
    """
    Curled Wake model wind farm solver. This solver needs a slightly different
    __call__ method because there are no "wakes" or "wake superposition" in the
    Curled Wake model. The wind farm is solved in a single step by numerically
    integrating a parabolic PDE.

    Follows the general framework of Martínez-Tossas et al. (2019, 2021).
    """

    def __init__(
        self,
        rotor_model: Optional[Rotor] = None,
        base_windfield: Optional[Windfield] = None,
        TIamb: float = None,
        solver_kwargs: Optional[dict] = None,
    ):
        """
        Initializes the CurledWindFarm. 
        
        Note that TIamb is unused. Instead, ensure that the `base_windfield` 
        includes ambient turbulence. 
        """
        self.rotor_model = AD() if rotor_model is None else rotor_model
        self.base_windfield = (
            Uniform(TIamb=TIamb) if base_windfield is None else base_windfield
        )
        self.TIamb = TIamb
        self.solver_kwargs = dict() if solver_kwargs is None else solver_kwargs

    def __call__(
        self, layout: Layout, setpoints: list[tuple[float, ...]]
    ) -> WindfarmSolution:
        """
        Solves for the rotor solutions in the wind farm layout, marching
        the CurledWakeWindfield to each rotor location as necessary.
        """
        # this function has to: 1) arrange rotors to march downstream, 2) initialize the CurledWakeWindfield, 3) solve, 4) aggregate results
        N = layout.x.size
        wakes = N * [None]  # this just remains as [None, ...] in CWM
        rotor_solutions = N * [None]

        windfield = CurledWakeWindfield(self.base_windfield, **self.solver_kwargs)

        for i, (x, y, z) in layout.iter_downstream():
            windfield.march_to(x=x, y=y, z=z)  # march to the next rotor location, extrapolating where necessary
            rotor_solutions[i] = self.rotor_model(x, y, z, windfield, *setpoints[i])
            rotor_solutions[i].idx = i
            windfield.stamp_ic(rotor_solutions[i], x, y, z)  # stamp the rotor solution into the windfield

        return WindfarmSolution(layout, setpoints, rotor_solutions, wakes, windfield)
