from tqdm import tqdm
from sklearn.metrics import classification_report
import torch

from src.trainer.trainer import Trainer

class TweetTrainer(Trainer):
    def __init__(self, get_model_func, configs, device, logger) -> None:
        super().__init__(get_model_func, configs, device, logger)

    def train(self, train_dataloader):
        self.model.train()
        total_loss = 0

        for batch in tqdm(train_dataloader):
            _, loss = self.model(batch)
            total_loss += loss

        return total_loss / len(train_dataloader)

    def eval(self, eval_dataloader):
        self.model.eval()
        actual_labels = []
        predicted_labels = []
        for batch in tqdm(eval_dataloader):
            pred, _ = self.model(batch)
            actual_labels.extend(batch['label_sexist'])
            predicted_labels.extend(pred)
        
        scores = classification_report(actual_labels, predicted_labels, output_dict=True)
        return scores, {}

    def predict(self, dataset):
        pass

    def summarize_scores(self, scores):
        return scores['macro avg']['f1-score']
