from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from tqdm import tqdm

from src.optimizer.utils import get_optimizer
from src.datasets.dataset import EDOSDataset
from src.trainer.edos_trainer import EDOSTrainer
from src.models.bert import BertClassifier


class SemiSupervisedLearning:
    def __init__(self, configs, state_configs, ground_truth_dataset, unlabeled_dataset, validation_dataset, logger, device) -> None:
        self.ground_truth_dataset = ground_truth_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.validation_dataset = validation_dataset
        self.predicted_datasets = []
        self.logger = logger
        self.device = device
        self.configs = configs
        self.state_configs = state_configs

    def run(self, model):
        prediction_model = model
        for fold in range(self.configs.ssl.k_fold):
            _, unlabeled_dataset = self.unlabeled_dataset.get_kth_fold_dataset(fold)
            unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=self.configs.train.eval_batch_size, shuffle=False)
            
            pred_data = []
            for batch in tqdm(unlabeled_dataloader):
                pred, _ = prediction_model(batch)
                for i, rewire_id in enumerate(batch['rewire_id']):
                    pred_data.append({
                        'rewire_id': rewire_id,
                        'text': batch['text'][i],
                        'label_sexist': pred[rewire_id]['sexist'],
                        'label_category': 'none',
                        'label_vector': 'none',
                        'confidence': pred[rewire_id]['confidence_s']['sexist']
                    })

            positive_pred_data = [data for data in pred_data if data['label_sexist'] == 'sexist']
            sorted_pred_data = sorted(positive_pred_data, key=lambda x: x['confidence'], reverse=True)
            
            # delete confidence key
            for data in sorted_pred_data:
                del data['confidence']


            predicted_dataset = EDOSDataset(f'pred_{fold}', self.configs, sorted_pred_data[:self.configs.ssl.additions], 6)
            self.predicted_datasets.append(predicted_dataset)

            ssl_train_datasets = [self.ground_truth_dataset]
            for pred_dataset in self.predicted_datasets:
                ssl_train_datasets.append(pred_dataset)

            train_dataset = ConcatDataset(ssl_train_datasets)
            train_dataloader = DataLoader(train_dataset, batch_size=self.configs.train.train_batch_size, shuffle=True)
            eval_dataloader = DataLoader(self.validation_dataset, batch_size=self.configs.train.eval_batch_size, shuffle=False)
            model = BertClassifier(self.configs, self.device)
            optimizer = get_optimizer(model, self.configs)
            trainer = EDOSTrainer(self.configs, self.state_configs, model, train_dataloader, eval_dataloader, optimizer, self.device, self.logger)
            trainer.run()

            prediction_model = trainer.get_best_model()
