from transformers import Adafactor
from transformers.optimization import AdafactorSchedule

from src.optimizer.optimizer import Optimizer


class AdaFactorOptimizer(Optimizer):
    def step_scheduler(self) -> None:
        pass

    def get_lr(self) -> None:
        pass

    def __init__(self, model, lr=None) -> None:
        self.optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=lr)
        self.scheduler = AdafactorSchedule(self.optimizer)

    def step_optimizer(self) -> None:
        self.optimizer.step()
        self.scheduler.step()

    def step_scheduler(self) -> None:
        self.scheduler.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()
