import os

import torch
import random

from src.config_reader import read_json_configs, read_dict_configs
from src.datasets.dataset import TrainDataset
from src.logger import Logger
from src.strategies.training.ssl.ssl_v1 import SemiSupervisedLearning
from src.strategies.ensemble.utils import get_ensemble_model
from src.utils import get_args

if __name__ == '__main__':
    args = get_args()
    configs = read_json_configs(os.path.join('./configs', args.config))

    logger = Logger(configs)

    torch.manual_seed(configs.seed)
    random.seed(configs.seed)

    state_configs_path = os.path.join(logger.dir, configs.logs.files.state)
    if os.path.exists(state_configs_path):
        state_configs = read_json_configs(state_configs_path)
    else:
        state_configs = read_dict_configs({'kth_fold': 0,
                                           'epoch': 0,
                                           'best_score': None,
                                           'epochs_without_improvement': 0,
                                           'kth_fold_metrics': []})

    train_dataset = TrainDataset('train', configs.train.files.train, configs.train.task, configs.train.k_fold)
    eval_dataset = TrainDataset('eval', configs.train.files.eval, configs.train.task, configs.train.k_fold)
    unlabelled_dataset = TrainDataset('unlabeled', configs.train.files.unlabeled, configs.train.task, configs.train.k_fold)

    ssl = SemiSupervisedLearning(configs, state_configs, train_dataset, eval_dataset, unlabelled_dataset, logger, args.device)
    ssl.run()
    print("Finished training with Semi supervised learning")
