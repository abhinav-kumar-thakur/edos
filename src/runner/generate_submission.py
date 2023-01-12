import os
from argparse import ArgumentParser
from collections import defaultdict, Counter

import torch
from tqdm import tqdm

from src.config_reader import read_json_configs
from src.models.utils import get_model
from src.logger import Logger
from src.datasets.dataset import PredictionDataset, DevDataset
from src.utils import get_args
from src.strategies.ensemble.utils import get_ensemble_model
from src.strategies.ensemble.voting import Voting


if __name__ == '__main__':
    args = get_args()
    configs = read_json_configs(os.path.join('./configs', args.config))

    logger = Logger(configs)
    dev_data = DevDataset(configs)
    
    voting_model = get_ensemble_model(configs, logger, args.device)
    prediction_dataloader = torch.utils.data.DataLoader(dev_data, batch_size=configs.predict.batch_size, shuffle=False, num_workers=0)
    
    predictions = {}
    for batch in tqdm(prediction_dataloader):
        pred, _ = voting_model.forward(batch, train=False)
        for rew_id, labels in pred.items():
            predictions[rew_id] = labels

    with open(os.path.join(logger.dir, configs.logs.files.submission), 'w') as f:
        f.write('rewire_id,label_pred\n')
        for rew_id, pred in predictions.items():
            f.write(f'{rew_id},{pred}\n')

    print("Done generating submission file!")
