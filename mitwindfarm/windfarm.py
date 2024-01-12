from numpy.typing import ArrayLike
from typing import Type

from .Layout import Layout
from .Rotor import Rotor
from .Windfield import Windfield
from .Wake import Wake
from .Superposition import Superposition


class WindfarmSolution:
    ...


class Windfarm:
    def __init__(
        self,
        turbines: list[Rotor],
        wake_model: Type[Wake],
        superposition: Superposition,
        base_windfield: Windfield,
    ):
        self.turbines = turbines
        self.wake_model = wake_model
        self.superposition = superposition
        self.base_windfield = base_windfield
        self.N = len(turbines)

    def __call__(self, layout: Layout, setpoints: list[tuple[float, ...]]) -> WindfarmSolution:
        wakes = self.N * [None]
        windfield = self.superposition(self.base_windfield, [])
        for i, (x, y, z) in layout.iter_downstream():
            rotor: Rotor = self.turbines[i]
            rotor_sol = rotor(*setpoints[i])
            wakes[i] = self.wake_model(x, y, z, rotor_sol)
            windfield.add_wake(wakes[i])

            raise NotImplementedError
