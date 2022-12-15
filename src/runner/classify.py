import os
from argparse import ArgumentParser

import torch
import random

from src.config_reader import read_json_configs
from src.models.utils import get_classification_model
from src.logger import Logger
from src.trainer.tweet_trainer import TweetTrainer


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--config', default='dev.json', required=True, help='Config file from ./configs')
    arg_parser.add_argument('--device', default='cpu', required=True, help='Supported devices: mps/cpu/cuda')

    args = arg_parser.parse_args()

    configs = read_json_configs(os.path.join('./configs', args.config))
    logger = Logger(configs)

    torch.manual_seed(configs.seed)
    random.seed(configs.seed)
    
    trainer = TweetTrainer(get_classification_model, configs, args.device, logger)
    trainer.train_kfold()

    print("Done loading dataset")
