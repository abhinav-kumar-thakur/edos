from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset

from src.optimizer.utils import get_optimizer
from src.trainer.edos_trainer import EDOSTrainer


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
            unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=self.configs.ssl.unlabeled_batch_size, shuffle=False)
            pred_data = []
            for batch in unlabeled_dataloader:
                pred_data.append(prediction_model.predict(batch))

            self.predicted_datasets.append(pred_data)

            train_dataset = ConcatDataset([self.ground_truth_dataset, pred_data])
            train_dataloader = DataLoader(train_dataset, batch_size=self.configs.ssl.labeled_batch_size, shuffle=True)
            eval_dataloader = DataLoader(self.validation_dataset, batch_size=self.configs.ssl.validation_batch_size, shuffle=False)
            optimizer = get_optimizer(model, self.configs)
            trainer = EDOSTrainer(self.configs, self.state_configs, model, train_dataloader, eval_dataloader, optimizer, self.device, self.logger)
            trainer.run()

            prediction_model = trainer.get_best_model()
