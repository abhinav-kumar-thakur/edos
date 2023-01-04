import os

from torch.utils.data import DataLoader, ConcatDataset

from src.config_reader import write_json_configs
from src.models.utils import get_classification_model_from_state
from src.optimizer.utils import get_optimizer
from src.trainer.edos_trainer import EDOSTrainer


class CrossValidation:
    def __init__(self, configs, state_configs, dataset, additional_dataset, logger, device):
        self.configs = configs
        self.state_configs = state_configs
        self.logger = logger
        self.device = device
        self.dataset = dataset
        self.additional_dataset = additional_dataset

    def run(self):
        for kth_fold in range(self.state_configs.kth_fold, self.configs.train.k_fold):
            if kth_fold not in self.configs.train.run_folds:
                print(f'Fold {kth_fold} is not in run_folds, skipping')
                self.state_configs.edit('epoch', 0)
                self.state_configs.edit('kth_fold', kth_fold + 1)
                continue

            train_set, eval_set = self.dataset.get_kth_fold_dataset(kth_fold)
            self.logger.log_file(self.configs.logs.files.data, train_set.summarize())
            self.logger.log_file(self.configs.logs.files.data, eval_set.summarize())

            model = get_classification_model_from_state(self.configs, self.state_configs, self.device)
            train_dataloader = DataLoader(ConcatDataset([train_set,self.additional_dataset]) , batch_size=self.configs.train.train_batch_size, shuffle=True)
            eval_dataloader = DataLoader(eval_set, batch_size=self.configs.train.eval_batch_size, shuffle=False)
            optimizer = get_optimizer(model, self.configs)
            trainer = EDOSTrainer(self.configs, self.state_configs, model, train_dataloader, eval_dataloader, optimizer, self.device, self.logger)
            trainer.run()

            self.state_configs.edit('epoch', 0)
            self.state_configs.edit('kth_fold', kth_fold + 1)
            self.state_configs.kth_fold_metrics.append(self.state_configs.best_score)
            write_json_configs(self.state_configs, os.path.join(self.logger.dir, self.configs.logs.files.state))

        kfold_eval_metrics = self.state_configs.kth_fold_metrics
        self.logger.log_file(self.configs.logs.files.best, {'avg': sum(kfold_eval_metrics) / len(kfold_eval_metrics), 'kfold_eval_metrics': kfold_eval_metrics})
