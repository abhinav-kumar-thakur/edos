from abc import abstractmethod, ABC
import json
import os

import torch

from src.config_reader import write_json_configs
from src.logger import Logger
from src.models.utils import save_model, get_model


class Trainer(ABC):
    def __init__(self, configs, state_configs, model, train_dataloader, eval_dataloader, optimizer, device, logger) -> None:
        self.state_configs = state_configs
        self.configs = configs
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.device = device
        self.logger: Logger = logger
        self.model_save_dir = os.path.join(configs.logs.dir, configs.title + '-' + configs.task, configs.logs.files.models)

    def run(self):
        if self.state_configs.epoch == 0:
            self.state_configs.edit('best_score', None)
            self.state_configs.edit('epochs_without_improvement', 0)

        for epoch in range(self.state_configs.epoch, self.configs.train.epochs):
            self.logger.log(f'Kth Fold: {self.state_configs.kth_fold}, Epoch: {epoch}')

            if self.state_configs.epochs_without_improvement >= self.configs.train.patience:
                break

            avg_loss = self.train(self.train_dataloader)
            train_scores, _ = self.eval(self.train_dataloader)
            eval_scores, eval_predictions = self.eval(self.eval_dataloader)
            eval_metric = self.summarize_scores(eval_scores)
            if self.state_configs.best_score is None or eval_metric > self.state_configs.best_score:
                self.state_configs.edit('best_score', eval_metric)
                best_params = {'kth_fold': self.state_configs.kth_fold, 'epoch': epoch, 'eval_metric': eval_scores}

                torch.save(self.model.state_dict(), os.path.join(self.model_save_dir, f'best_model_{self.state_configs.kth_fold}.pt'))
                json.dump(best_params, open(os.path.join(self.model_save_dir, f'best_metric_{self.state_configs.kth_fold}.json'), 'w'))

                self.logger.log_file(self.configs.logs.files.best, best_params)
                self.logger.log_csv(f'{self.state_configs.kth_fold}_{epoch}_{self.configs.logs.files.predictions}', eval_predictions)
                self.state_configs.edit('epochs_without_improvement', 0)
            else:
                self.state_configs.edit('epochs_without_improvement', self.state_configs.epochs_without_improvement + 1)

            train_scores['loss'] = avg_loss
            self.logger.log_file(self.configs.logs.files.train, {"Kth Fold": self.state_configs.kth_fold, "Epoch": epoch, 'train': train_scores})
            self.logger.log_file(self.configs.logs.files.train, {"Kth Fold": self.state_configs.kth_fold, "Epoch": epoch, 'eval': eval_scores})
            self.logger.log(f'Train Score: {train_scores} \n\n Eval Score: {eval_scores} \n\n Best Score: {self.state_configs.best_score} \n\n Epochs Without Improvement: {self.state_configs.epochs_without_improvement}')
            save_model(self.model, self.configs)
            self.state_configs.edit('epoch', epoch + 1)
            write_json_configs(self.state_configs, os.path.join(self.logger.dir, self.configs.logs.files.state))

    def get_best_model(self):
        model_path = os.path.join(self.model_save_dir, f'best_model_{self.state_configs.kth_fold}.pt')
        return get_model(self.configs, model_path, self.device)

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
