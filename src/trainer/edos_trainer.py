from sklearn.metrics import classification_report
from tqdm import tqdm

from src.trainer.trainer import Trainer


class EDOSTrainer(Trainer):
    def __init__(self, configs, state_configs, model, train_dataloader, eval_dataloader, optimizer, device, logger) -> None:
        super().__init__(configs, state_configs, model, train_dataloader, eval_dataloader, optimizer, device, logger)

    def train(self, train_dataloader):
        self.model.train()
        total_loss = 0

        for batch in tqdm(train_dataloader):
            self.optimizer.zero_grad()
            _, loss = self.model(batch)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step_optimizer()

        self.optimizer.step_scheduler()
        return total_loss / len(train_dataloader)

    def eval(self, eval_dataloader):
        self.model.eval()
        actual_a, actual_b, actual_c = [], [], []
        predicted_a, predicted_b, predicted_c = [], [], []
        predictions = [('rewire_id', 'text', 'pred_a', 'pred_b', 'pred_c', 'actual_a', 'actual_b', 'actual_c')]
        for batch in tqdm(eval_dataloader):
            pred, _ = self.model(batch, train=False)
            for i, rewire_id in enumerate(batch['rewire_id']):
                predictions.append((
                    rewire_id, batch['text'][i],
                    pred[rewire_id]['sexist'],
                    pred[rewire_id]['category'],
                    pred[rewire_id]['vector'],
                    batch['label_sexist'][i],
                    batch['label_category'][i],
                    batch['label_vector'][i]
                ))

                actual_a.append(batch['label_sexist'][i])
                predicted_a.append(pred[rewire_id]['sexist'])

                if batch['label_sexist'][i] == 'sexist' and 'b' in self.configs.train_config['task'] and 'c' in self.configs.train_config['task']:

                    actual_b.append(batch['label_category'][i])
                    actual_c.append(batch['label_vector'][i])

                    predicted_b.append(pred[rewire_id]['category'])
                    predicted_c.append(pred[rewire_id]['vector'])

        scores_a = classification_report(actual_a, predicted_a, output_dict=True)
        if 'b' in self.configs.train_config['task'] and 'c' in self.configs.train_config['task']:   
            scores_b = classification_report(actual_b, predicted_b, output_dict=True)
            scores_c = classification_report(actual_c, predicted_c, output_dict=True)

        scores = {
            'a': scores_a,
            'b': scores_b,
            'c': scores_c
        }

        return scores, predictions

    def predict(self, dataset):
        pass

    def summarize_scores(self, scores):
        return scores['a']['macro avg']['f1-score']
