from torch.optim import Adam

from src.optimizer.optimizer import Optimizer


class AdamOptimizer(Optimizer):
    def __init__(self, model, lr=0.00001) -> None:
        self.optimizer = Adam(model.get_trainable_parameters(), lr=lr)

    def step(self) -> None:
        self.optimizer.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()
