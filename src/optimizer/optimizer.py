from abc import ABC, abstractmethod


class Optimizer(ABC):
    @abstractmethod
    def step_optimizer(self) -> None:
        pass

    @abstractmethod
    def zero_grad(self) -> None:
        pass

    @abstractmethod
    def step_scheduler(self) -> None:
        pass
    
    @abstractmethod
    def get_lr(self) -> None:
        pass
