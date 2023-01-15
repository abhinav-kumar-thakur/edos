from transformers import AdamW, get_cosine_schedule_with_warmup

from .optimizer import Optimizer
from .scheduler import get_scheduler


class AdamWOptimizer(Optimizer):
    def __init__(self, model, configs) -> None:
        self.optimizer = AdamW(model.get_trainable_parameters(), lr=configs.train.optimizer.lr, weight_decay=configs.train.optimizer.weight_decay)
        self.scheduler = get_scheduler(self.optimizer, configs)

    def step_optimizer(self):
        self.optimizer.step()

    def step_scheduler(self) -> None:
        self.scheduler.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_lr(self):
        return self.scheduler.get_last_lr()
