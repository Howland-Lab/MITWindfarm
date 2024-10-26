from abc import ABC, abstractmethod

from .Wake import Wake
from .Windfield import Superimposed, Windfield


class Superposition(ABC):
    @abstractmethod
    def __call__(self, base_windfield: Windfield, wakes: list[Wake]) -> Windfield:
        ...


class Linear(Superposition):
    def __call__(self, base_windfield: Windfield, wakes: list[Wake]) -> Windfield:
        return Superimposed(base_windfield, wakes, method="linear")

class Niayifar(Superposition):
    def __call__(self, base_windfield: Windfield, wakes: list[Wake]) -> Windfield:
        return Superimposed(base_windfield, wakes, method="niayifar")


class Quadratic(Superposition):
    def __call__(self, base_windfield: Windfield, wakes: list[Wake]) -> Windfield:
        return Superimposed(base_windfield, wakes, method="quadratic")


class Dominant(Superposition):
    def __call__(self, base_windfield: Windfield, wakes: list[Wake]) -> Windfield:
        return Superimposed(base_windfield, wakes, method="dominant")
