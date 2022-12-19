from abc import abstractmethod, ABC
import os

import torch
from torch.utils.data import DataLoader

from src.logger import Logger
from src.datasets.dataset import TrainDataset


class Trainer(ABC):
    def __init__(self, get_model_func, configs, device, logger) -> None:
        self.get_model_func = get_model_func
        self.configs = configs
        self.model = None
        self.optimizer = None
        self.train_dataset = TrainDataset(configs)
        self.device = device
        self.logger: Logger = logger
        self.model_save_dir = os.path.join(configs.logs.dir, configs.title + '-' + configs.task, configs.logs.files.models)

    def train_kfold(self):
        state = self.logger.get_state()
        current_fold = 0
        current_epoch = 0
        
        if state['kth_fold'] is not None:
            self.model = self.get_model_func(self.configs, self.device)
            self.model.load_state_dict(self.logger.get_current_state_model())
            current_fold = state['kth_fold'] + 1
            current_epoch = state['epoch'] + 1
        else:
            self.logger.log_file(self.configs.logs.files.data, self.train_dataset.summarize())

        for kth_fold in range(current_fold, self.configs.train.k_fold):
            self.model = self.get_model_func(self.configs, self.device)
            if current_epoch > 0:
                self.model.load_state_dict(self.logger.get_current_state_model())

            train_set, eval_set = self.train_dataset.get_kth_fold_dataset(kth_fold)
            self.logger.log_file(self.configs.logs.files.data, train_set.summarize())
            self.logger.log_file(self.configs.logs.files.data, eval_set.summarize())

            train_dataloader = DataLoader(train_set, batch_size=self.configs.train.train_batch_size, shuffle=True)
            eval_dataloader = DataLoader(eval_set, batch_size=self.configs.train.eval_batch_size, shuffle=False)

            best_score = None
            best_parames = {}
            epcohs_without_improvement = 0

            for epoch in range(current_epoch, self.configs.train.epochs):
                avg_loss = self.train(train_dataloader)
                train_scores, train_predictions = self.eval(train_dataloader)
                eval_scores, eval_predictions = self.eval(eval_dataloader)
                eval_metric = self.summarize_scores(eval_scores)
                if best_score is None or  eval_metric > best_score:
                    best_score = eval_metric
                    best_parames = {
                        'kth_fold': kth_fold,
                        'epoch': epoch,
                        'eval_metric': eval_scores,
                    }

                    torch.save(self.model.state_dict(), os.path.join(self.model_save_dir, f'best_model_{kth_fold}.pt'))
                    self.logger.log_file(self.configs.logs.files.best, best_parames)
                    self.logger.log_file(self.configs.logs.files.predictions, {'kth_fold': kth_fold, 'epoch': epoch, 'eval': eval_predictions})
                    epcohs_without_improvement = 0
                else:
                    epcohs_without_improvement += 1
                
                self.logger.log_file(self.configs.logs.files.train, {"Kth Fold": kth_fold, "Epoch": epoch, 'train': train_scores, 'loss': avg_loss})
                self.logger.log_file(self.configs.logs.files.train, {"Kth Fold": kth_fold, "Epoch": epoch, 'eval': eval_scores})

                if epcohs_without_improvement >= self.configs.train.patience:
                    break
            
            self.logger.update_eval_metrics(best_score)
            
        kfold_eval_metrics = self.logger.get_eval_metrics()
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