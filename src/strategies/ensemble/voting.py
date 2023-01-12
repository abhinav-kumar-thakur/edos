import os
import itertools
from collections import defaultdict, Counter

import torch
from tqdm import tqdm

from src.logger import Logger
from src.datasets.dataset import DevDataset
from src.models.utils import get_model
from .ensemble import Ensemble

class Voting(Ensemble, torch.nn.Module):
    def __init__(self, config, logger, device='cpu'):
        self.device = device
        self.models = []
        self.configs = config
        self.logger = logger
        self.model_dir = os.path.join(self.logger.dir, self.configs.logs.files.models)
        for file in os.listdir(self.model_dir):
            if 'best_model' in file:
                self.models.append(file)
        
    def forward(self, batch, train=False):
        predictions = defaultdict(list)
        logger = Logger(self.configs)

        for model_path in self.models:
            model = get_model(self.configs, os.path.join(self.model_dir, model_path), 'cpu')
            model.eval()
            pred, loss = model(batch, train=False)
            for rewire_id in batch['rewire_id']:
                predictions[rewire_id].append(pred[rewire_id]['sexist'])

        labels = {}
        for rew_id, pred in predictions.items():
            label = Counter(pred).most_common(1)[0][0]
            labels[rew_id] = label

        return labels, None

        