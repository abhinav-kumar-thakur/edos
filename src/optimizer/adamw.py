from torch.optim import AdamW

from src.optimizer.optimizer import Optimizer


class AdamWOptimizer(Optimizer):
    def __init__(self, model, lr) -> None:
        self.optimizer = AdamW(model.get_trainable_parameters(), lr=lr)

    def step(self) -> None:
        self.optimizer.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()