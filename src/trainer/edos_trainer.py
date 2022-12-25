from tqdm import tqdm
from sklearn.metrics import classification_report

from src.optimizer.utils import get_optimizer
from src.trainer.trainer import Trainer


class EDOSTrainer(Trainer):
    def __init__(self, configs, state_configs, model, train_dataloader, eval_dataloader, device, logger) -> None:
        super().__init__(configs, state_configs, model, train_dataloader, eval_dataloader, device, logger)

    def train(self, train_dataloader):
        self.model.train()
        total_loss = 0
        optimizer = get_optimizer(self.model, self.configs)

        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            _, loss = self.model(batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        return total_loss / len(train_dataloader)

    def eval(self, eval_dataloader):
        self.model.eval()
        actual_labels = []
        predicted_labels = []
        predictions = [('input', 'pred', 'actual')]
        for batch in tqdm(eval_dataloader):
            pred, _ = self.model(batch, train=False)
            actual_labels.extend(batch['label_sexist'])
            predicted_labels.extend(pred)
            predictions.extend(zip(batch['text'], pred, batch['label_sexist']))

        scores = classification_report(actual_labels, predicted_labels, output_dict=True)
        return scores, predictions

    def predict(self, dataset):
        pass

    def summarize_scores(self, scores):
        return scores['macro avg']['f1-score']
