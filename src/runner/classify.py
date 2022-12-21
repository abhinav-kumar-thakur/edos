import os
from argparse import ArgumentParser

import torch
import random

from src.config_reader import read_json_configs, read_dict_configs
from src.logger import Logger
from src.trainer.tweet_trainer import TweetTrainer


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--config', default='dev.json', required=True, help='Config file from ./configs')
    arg_parser.add_argument('--device', default='cpu', required=True, help='Supported devices: mps/cpu/cuda')

    args = arg_parser.parse_args()

    configs = read_json_configs(os.path.join('./configs', args.config))

    logger = Logger(configs)
    state_configs_path = os.path.join(logger.dir, configs.logs.files.state)
    if os.path.exists(state_configs_path):
        state_configs = read_json_configs(state_configs_path)
    else:
        state_configs = read_dict_configs({'kth_fold': 0,
                                           'epoch': 0,
                                           'best_score': None,
                                           'epochs_without_improvement': 0, 'kth_fold_metrics': []})

    torch.manual_seed(configs.seed)
    random.seed(configs.seed)

    trainer = TweetTrainer(state_configs, configs, args.device, logger)
    trainer.train_kfold()

    print("Done loading dataset")
