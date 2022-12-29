import os
from argparse import ArgumentParser
from collections import defaultdict, Counter

import torch
from tqdm import tqdm

from src.config_reader import read_json_configs
from src.models.utils import get_model
from src.logger import Logger
from src.datasets.dataset import PredictionDataset
from src.utils import get_args


if __name__ == '__main__':
    args = get_args()
    configs = read_json_configs(os.path.join('./configs', args.config))

    logger = Logger(configs)
    model_dir = os.path.join(logger.dir, configs.logs.files.models)
    models = []
    for file in os.listdir(model_dir):
        if 'best_model' in file:
            models.append(file)

    prediction_dataset = PredictionDataset(configs)
    prediction_dataloader = torch.utils.data.DataLoader(prediction_dataset, batch_size=configs.predict.batch_size, shuffle=False, num_workers=0)
    
    predictions = defaultdict(list)
    for model_path in models:
        model = get_model(configs, os.path.join(model_dir, model_path), args.device)
        model.eval()
        for batch in tqdm(prediction_dataloader):
            pred, loss = model(batch, train=False)
            for rewire_id, pred in zip(batch['rewire_id'], pred):
                predictions[rewire_id].append(pred)

    with open(os.path.join(logger.dir, configs.logs.files.submission), 'w') as f:
        f.write('rewire_id,label_pred\n')
        for rew_id, pred in predictions.items():
            label = Counter(pred).most_common(1)[0][0]
            f.write(f'{rew_id},{label}\n')

    print("Done generating submission file!")
