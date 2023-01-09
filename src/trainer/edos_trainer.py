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
        predictions = [(
            'rewire_id', 'text',
            'actual_a', 'pred_a', 'confidence_a', 'uncertainity_a',
            'actual_b', 'pred_b', 'confidence_b', 'uncertainity_b',
            'actual_c', 'pred_c', 'confidence_c', 'uncertainity_c'
        )]
        for batch in tqdm(eval_dataloader):
            pred, _ = self.model(batch, train=False)
            for i, rewire_id in enumerate(batch['rewire_id']):
                predictions.append((
                    rewire_id, batch['text'][i],
                    batch['label_sexist'][i],
                    pred[rewire_id]['sexist'] if 'a' in self.configs.train.task else '-',
                    pred[rewire_id]['confidence']['sexist'] if 'a' in self.configs.train.task else '-',
                    pred[rewire_id]['uncertainity']['sexist'] if 'a' in self.configs.train.task else '-',
                    batch['label_category'][i],
                    pred[rewire_id]['category'] if 'b' in self.configs.train.task else '-',
                    pred[rewire_id]['confidence']['category'] if 'b' in self.configs.train.task else '-',
                    pred[rewire_id]['uncertainity']['category'] if 'b' in self.configs.train.task else '-',
                    batch['label_vector'][i],
                    pred[rewire_id]['vector'] if 'c' in self.configs.train.task else '-',
                    pred[rewire_id]['confidence']['vector'] if 'c' in self.configs.train.task else '-',
                    pred[rewire_id]['uncertainity']['vector'] if 'c' in self.configs.train.task else '-'
                ))

                if 'a' in self.configs.train.task:
                    actual_a.append(batch['label_sexist'][i])
                    predicted_a.append(pred[rewire_id]['sexist'])

                if batch['label_sexist'][i] == 'sexist' and 'b' in self.configs.train.task:
                    actual_b.append(batch['label_category'][i])
                    predicted_b.append(pred[rewire_id]['category'])

                if batch['label_sexist'][i] == 'sexist' and 'c' in self.configs.train.task:
                    actual_c.append(batch['label_vector'][i])
                    predicted_c.append(pred[rewire_id]['vector'])

        scores = {
            'a': classification_report(actual_a, predicted_a, output_dict=True) if 'a' in self.configs.train.task else None,
            'b': classification_report(actual_b, predicted_b, output_dict=True) if 'b' in self.configs.train.task else None,
            'c': classification_report(actual_c, predicted_c, output_dict=True) if 'c' in self.configs.train.task else None
        }

        return scores, predictions

    def predict(self, dataset):
        pass

    def summarize_scores(self, scores):
        score = 1
        if "a" in self.configs.train.task:
            score *= scores['a']['macro avg']['f1-score']
        
        if "b" in self.configs.train.task:
            score *= scores['b']['macro avg']['f1-score']
        
        if "c" in self.configs.train.task:
            score *= scores['c']['macro avg']['f1-score']

        return score