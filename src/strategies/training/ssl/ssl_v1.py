from collections import defaultdict
import os

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset

from src.logger import Logger
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
        self.logger: Logger = logger
        self.device = device
        self.configs = configs
        self.state_configs = state_configs
        self.ssl_dataset = None
        self.current_metric = 0
        self.current_model = None
        self.last_best_metric = 0
        self.prediction_model = None
        self.used_rewire_ids = set()

    def run(self):
        for fold in range(self.state_configs.kth_fold, self.configs.ssl.k_fold):
            model = get_classification_model_from_state(self.configs, self.state_configs, self.device)
            train_dataset = ConcatDataset([self.ground_truth_dataset, self.ssl_dataset]) if self.ssl_dataset else self.ground_truth_dataset
            train_dataloader = DataLoader(train_dataset, batch_size=self.configs.train.train_batch_size, shuffle=True)
            eval_dataloader = DataLoader(self.validation_dataset, batch_size=self.configs.train.eval_batch_size, shuffle=False)

            optimizer = get_optimizer(model, self.configs)
            trainer = EDOSTrainer(self.configs, self.state_configs, model, train_dataloader, eval_dataloader, optimizer, self.device, self.logger)
            trainer.run()

            self.current_metric = trainer.get_best_metric()['eval_metric']['b']['macro avg']['f1-score']
            self.current_model = trainer.get_best_model()
        
            self.prediction_model = self.current_model if self.current_metric > self.last_best_metric else self.prediction_model
            self.last_best_metric = self.current_metric if self.current_metric > self.last_best_metric else self.last_best_metric
            unlabeled_dataloader = DataLoader(self.unlabeled_dataset, batch_size=self.configs.train.eval_batch_size, shuffle=False)

            pred_data = []
            for batch in tqdm(unlabeled_dataloader):
                pred, _ = self.prediction_model(batch, train=False)
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
                if data['rewire_id'] not in self.used_rewire_ids:
                    label_wise_pred_data[data['label_category']].append(data)

            selected_data = []
            for label, data in label_wise_pred_data.items():
                sorted_data = sorted(data, key=lambda x: x['confidence'], reverse=True)
                selected_data += sorted_data[:self.configs.ssl.num_samples_per_label]

            # Update used rewire ids
            self.used_rewire_ids.update([data['rewire_id'] for data in selected_data])
            
            # delete confidence key
            for data in selected_data:
                del data['confidence']
            
            self.ssl_dataset = EDOSDataset(f'pred_{fold}', selected_data, 6)
            print(self.ssl_dataset.summarize())
            
            self.state_configs.edit('epoch', 0)
            self.state_configs.edit('kth_fold', fold + 1)
            self.state_configs.edit('best_score', 0)
            self.state_configs.edit('epochs_without_improvement', 0)
            self.state_configs.kth_fold_metrics.append(self.state_configs.best_score)
            write_json_configs(self.state_configs, os.path.join(self.logger.dir, self.configs.logs.files.state))
            self.logger.log_console(f'Fold {fold} completed: best metric: {self.last_best_metric} current metric: {self.current_metric}')
