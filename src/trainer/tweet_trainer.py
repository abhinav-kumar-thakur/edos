from tqdm import tqdm
from sklearn.metrics import classification_report
import torch

from src.trainer.trainer import Trainer

class TweetTrainer(Trainer):
    def __init__(self, get_model_func, configs, device, logger) -> None:
        super().__init__(get_model_func, configs, device, logger)

        class_weights = torch.FloatTensor([1, 3]).to(device)
        self.loss = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.label2idx = self.configs.datasets.label_sexist_ids.configs


    def train(self, train_dataloader):
        self.model.train()
        total_loss = 0

        for batch in tqdm(train_dataloader):
            self.optimizer.zero_grad()
            pred = self.model(batch)
            actaul = [self.label2idx[l] for l in  batch['label_sexist']]
            loss = self.loss(pred, torch.tensor(actaul).to(self.device))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            break

        return total_loss / len(train_dataloader)

    def eval(self, eval_dataloader):
        self.model.eval()
        actual_labels = []
        predicted_labels = []
        for batch in tqdm(eval_dataloader):
            pred = self.model(batch)
            pred = torch.argmax(pred, dim=1)
            predicted_labels.extend(pred.tolist())
            actual_labels.extend([self.label2idx[l] for l in  batch['label_sexist']])
        
        scores = classification_report(actual_labels, predicted_labels, output_dict=True)

        return scores, {}
        

    def predict(self, dataset):
        pass

    def summarize_scores(self, scores):
        return scores['macro avg']['f1-score']

    