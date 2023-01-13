import os

import torch
import random

from src.config_reader import read_json_configs, read_dict_configs
from src.datasets.dataset import TrainDataset, UnlabelledDataset, DevDataset
from src.logger import Logger
from src.strategies.training.ssl import SemiSupervisedLearning
from src.strategies.ensemble.utils import get_ensemble_model
from src.utils import get_args

if __name__ == '__main__':
    args = get_args()
    configs = read_json_configs(os.path.join('./configs', args.config))

    logger = Logger(configs)
    state_configs_path = os.path.join(logger.dir, configs.logs.files.state)
    if os.path.exists(state_configs_path):
        state_configs = read_json_configs(state_configs_path)
    else:
        state_configs = read_dict_configs({'kth_fold': 0,
                                           'epoch': 0,
                                           'best_score': None,
                                           'epochs_without_improvement': 0,
                                           'kth_fold_metrics': []})

    torch.manual_seed(configs.seed)
    random.seed(configs.seed)

    dataset = TrainDataset(configs)
    unlabelled_dataset = UnlabelledDataset(configs)
    validation_dataset = DevDataset(configs)

    ensemble_model = get_ensemble_model(configs, logger, args.device)
    ssl = SemiSupervisedLearning(configs, state_configs, dataset, unlabelled_dataset, validation_dataset, logger, args.device)
    ssl.run(ensemble_model)
    print("Finished training with Semi supervised learning")
