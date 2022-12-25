from abc import ABC, abstractmethod


class Optimizer(ABC):
    @abstractmethod
    def step(self) -> None:
        pass

    @abstractmethod
    def zero_grad(self) -> None:
        pass
