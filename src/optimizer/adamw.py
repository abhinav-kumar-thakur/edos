from transformers import AdamW, get_cosine_schedule_with_warmup

from src.optimizer.optimizer import Optimizer


class AdamWOptimizer(Optimizer):
    def __init__(self, model, configs) -> None:
        self.optimizer = AdamW(model.get_trainable_parameters(), lr=configs.train.optimizer.lr)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=configs.train.optimizer.num_warmup_steps, num_training_steps=configs.train.epochs)

    def step_optimizer(self):
        self.optimizer.step()

    def step_scheduler(self) -> None:
        self.scheduler.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_lr(self):
        return self.scheduler.get_last_lr()
