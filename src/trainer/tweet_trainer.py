from src.trainer.trainer import Trainer
import torch

class TweetTrainer(Trainer):
    def __init__(self, get_model_func, configs, device, logger) -> None:
        super().__init__(get_model_func, configs, device, logger)

    def train(self, train_dataloader):
        self.model.train()
        for batch in train_dataloader:
            self.optimizer.zero_grad()
            loss = self.model(batch)
            loss.backward()
            self.optimizer.step

    def eval(self, eval_dataloader):
        self.model.eval()
        scores = []
        predictions = []
        with torch.no_grad():
            for batch in eval_dataloader:
                score, prediction = self.model(batch)
                scores.append(score)
                predictions.append(prediction)
        return scores, predictions

    def predict(self, dataset):
        pass

    def summarize_scores(self, scores, class_distribution):
        return scores

    