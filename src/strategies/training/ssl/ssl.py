from collections import defaultdict
import os

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset

from src.config_reader import write_json_configs
from src.optimizer.utils import get_optimizer
from src.datasets.dataset import EDOSDataset
from src.trainer.edos_trainer import EDOSTrainer
from src.models.utils import get_classification_model_from_state


class SemiSupervisedLearning:
    def __init__(self, configs, state_configs, ground_truth_dataset, validation_dataset, unlabeled_dataset, logger, device) -> None:
        self.ground_truth_dataset = ground_truth_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.validation_dataset = validation_dataset
        self.logger = logger
        self.device = device
        self.configs = configs
        self.state_configs = state_configs
        self.ssl_datasets = []

    def run(self):
        for fold in range(self.configs.ssl.k_fold):
            model = get_classification_model_from_state(self.configs, self.state_configs, self.device)
            
            training_datasets = [self.ground_truth_dataset]
            if self.ssl_datasets:
                training_datasets += self.ssl_datasets

            train_dataset = ConcatDataset(training_datasets)
            train_dataloader = DataLoader(train_dataset, batch_size=self.configs.train.train_batch_size, shuffle=True)
            eval_dataloader = DataLoader(self.validation_dataset, batch_size=self.configs.train.eval_batch_size, shuffle=False)

            optimizer = get_optimizer(model, self.configs)
            trainer = EDOSTrainer(self.configs, self.state_configs, model, train_dataloader, eval_dataloader, optimizer, self.device, self.logger)
            trainer.run()

            best_metric = trainer.get_best_metric()['macro avg']['f1-score']
            prediction_model = trainer.get_best_model()

            last_best_metric = self.state_configs['kth_fold_metrics'][-1] if self.state_configs['kth_fold_metrics'] else 0
            if best_metric > last_best_metric:
                _, unlabeled_dataset = self.unlabeled_dataset.get_kth_fold_dataset(fold)
                unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=self.configs.train.eval_batch_size, shuffle=False)
                
                pred_data = []
                for batch in tqdm(unlabeled_dataloader):
                    pred, _ = prediction_model(batch, train=False)
                    for i, rewire_id in enumerate(batch['rewire_id']):
                        pred_data.append({
                            'rewire_id': rewire_id,
                            'text': batch['text'][i],
                            'label_sexist': pred[rewire_id]['sexist'] if pred[rewire_id]['sexist'] else 'unknown',
                            'label_category': pred[rewire_id]['category'],
                            'label_vector': pred[rewire_id]['vector'],
                            'confidence': pred[rewire_id]['confidence_s']['category']
                        })
                
                label_wise_pred_data = defaultdict(list)
                for data in pred_data:
                    label_wise_pred_data[data['label_category']].append(data)

                selected_data = []
                for label, data in label_wise_pred_data.items():
                    sorted_data = sorted(data, key=lambda x: x['confidence'], reverse=True)
                    selected_data += sorted_data[:self.configs.ssl.num_samples_per_label]

                # delete confidence key
                for data in selected_data:
                    del data['confidence']
                
                predicted_dataset = EDOSDataset(f'pred_{fold}', selected_data, 6)
                print(predicted_dataset.summarize())
                self.ssl_datasets.append(predicted_dataset)
            
            self.state_configs.edit('epoch', 0)
            self.state_configs.edit('kth_fold', fold + 1)
            self.state_configs.edit('best_score', 0)
            self.state_configs.edit('epochs_without_improvement', 0)
            self.state_configs.kth_fold_metrics.append(self.state_configs.best_score)
            write_json_configs(self.state_configs, os.path.join(self.logger.dir, self.configs.logs.files.state))
