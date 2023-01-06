import os

import torch
import random

from src.config_reader import read_json_configs, read_dict_configs
from src.datasets.dataset import TrainDataset, AdditionalTrainDataset, MamiDataset
from src.logger import Logger
from src.strategies.cross_validation import CrossValidation
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
    additional_datasets = [AdditionalTrainDataset(configs), MamiDataset(configs)]
    CrossValidation(configs, state_configs, dataset, additional_datasets, logger, args.device).run()

    print("Finished training with cross validation")
