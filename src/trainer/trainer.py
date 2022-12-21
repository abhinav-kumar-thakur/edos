from abc import abstractmethod, ABC
import os

import torch
from torch.utils.data import DataLoader

from src.config_reader import write_json_configs
from src.logger import Logger
from src.datasets.dataset import TrainDataset
from src.models.utils import get_classification_model_from_state, save_model


class Trainer(ABC):
    def __init__(self, state_configs, configs, device, logger) -> None:
        self.state_configs = state_configs
        self.configs = configs
        self.model = None
        self.train_dataset = TrainDataset(configs)
        self.device = device
        self.logger: Logger = logger
        self.model_save_dir = os.path.join(configs.logs.dir, configs.title + '-' + configs.task, configs.logs.files.models)

    def train_kfold(self):
        for kth_fold in range(self.state_configs.kth_fold, self.configs.train.k_fold):
            train_set, eval_set = self.train_dataset.get_kth_fold_dataset(kth_fold)
            self.logger.log_file(self.configs.logs.files.data, train_set.summarize())
            self.logger.log_file(self.configs.logs.files.data, eval_set.summarize())

            # oversample train set
            train_set.oversample_the_dataset()
            self.logger.log_file(self.configs.logs.files.data, train_set.summarize())


            self.model = get_classification_model_from_state(self.configs, self.state_configs, self.device)
            train_dataloader = DataLoader(train_set, batch_size=self.configs.train.train_batch_size, shuffle=True)
            eval_dataloader = DataLoader(eval_set, batch_size=self.configs.train.eval_batch_size, shuffle=False)

            if self.state_configs.epoch == 0:
                self.state_configs.edit('best_score', None)
                self.state_configs.edit('epochs_without_improvement', 0)
                
            for epoch in range(self.state_configs.epoch, self.configs.train.epochs):
                self.logger.log(f'Kth Fold: {kth_fold}, Epoch: {epoch}')

                avg_loss = self.train(train_dataloader)
                train_scores, train_predictions = self.eval(train_dataloader)
                eval_scores, eval_predictions = self.eval(eval_dataloader)
                eval_metric = self.summarize_scores(eval_scores)
                if self.state_configs.best_score is None or  eval_metric > self.state_configs.best_score:
                    self.state_configs.edit('best_score', eval_metric)
                    best_parames = {'kth_fold': kth_fold,'epoch': epoch,'eval_metric': eval_scores}

                    torch.save(self.model.state_dict(), os.path.join(self.model_save_dir, f'best_model_{kth_fold}.pt'))
                    self.logger.log_file(self.configs.logs.files.best, best_parames)
                    self.logger.log_csv(f'{self.configs.logs.files.predictions}_{kth_fold}', eval_predictions)
                    self.state_configs.edit('epochs_without_improvement', 0)
                else:
                    self.state_configs.edit('epochs_without_improvement', self.state_configs.epochs_without_improvement + 1)
                
                self.logger.log_file(self.configs.logs.files.train, {"Kth Fold": kth_fold, "Epoch": epoch, 'train': train_scores, 'loss': avg_loss})
                self.logger.log_file(self.configs.logs.files.train, {"Kth Fold": kth_fold, "Epoch": epoch, 'eval': eval_scores})
                self.logger.log(f'Train Score: {train_scores} \n\n Eval Score: {eval_scores} \n\n Best Score: {self.state_configs.best_score} \n\n Epochs Without Improvement: {self.state_configs.epochs_without_improvement}')
                save_model(self.model, self.configs)
                self.state_configs.edit('epoch', epoch + 1)
                write_json_configs(self.state_configs, os.path.join(self.logger.dir, self.configs.logs.files.state))

                
                if self.state_configs.epochs_without_improvement >= self.configs.train.patience:
                    break

            self.state_configs.edit('epoch', 0)
            self.state_configs.edit('kth_fold', kth_fold + 1)
            self.state_configs.kth_fold_metrics.append(self.state_configs.best_score)
            write_json_configs(self.state_configs, os.path.join(self.logger.dir, self.configs.logs.files.state))

        kfold_eval_metrics = self.state_configs.kth_fold_metrics    
        self.logger.log_file(self.configs.logs.files.best, {'avg': sum(kfold_eval_metrics)/len(kfold_eval_metrics), 'kfold_eval_metrics': kfold_eval_metrics})
        

    @abstractmethod
    def summarize_scores(self, scores):
        pass
        
    @abstractmethod
    def train(self, dataset):
        pass

    @abstractmethod
    def eval(self, dataset):
        pass

    @abstractmethod
    def predict(self, dataset):
        pass