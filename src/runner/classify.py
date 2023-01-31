import os

import torch
import random

from src.config_reader import read_json_configs, read_dict_configs
from src.datasets.dataset import TrainDataset, DevDataset
from src.logger import Logger
from src.strategies.training.cross_validation import CrossValidation
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

    train_dataset = TrainDataset('train', configs.train.files.train, configs.train.task, configs.train.k_fold)
    dev_dataset = DevDataset(configs)
    if 'a' not in configs.train.task:
        dev_dataset.data = [d for d in dev_dataset.data if d['label_sexist'] != 'sexist']

    dataset = train_dataset.merge(dev_dataset)

    additional_datasets = []
    CrossValidation(configs, state_configs, dataset, additional_datasets, logger, args.device).run()
    print("Finished training with cross validation")
