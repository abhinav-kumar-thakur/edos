import os
import itertools
from collections import defaultdict, Counter

import torch
from tqdm import tqdm

from src.logger import Logger
from src.datasets.dataset import DevDataset
from src.models.utils import get_model
from .ensemble import Ensemble

class Voting(Ensemble):
    def __init__(self, config, logger, device='cpu'):
        super().__init__()
        self.device = device
        self.models = []
        self.configs = config
        self.logger = logger
        self.model_dir = os.path.join(self.logger.dir, self.configs.logs.files.models)
        for file in os.listdir(self.model_dir):
            if 'best_model' in file:
                model = get_model(self.configs, os.path.join(self.model_dir, file), self.device)
                self.models.append(model)
        
    def forward(self, batch, train=False):
        predictions = defaultdict(list)
        for model in self.models:
            model.eval()
            pred, loss = model(batch, train=False)
            for rewire_id in batch['rewire_id']:
                predictions[rewire_id].append((pred[rewire_id]['sexist'], pred[rewire_id]['confidence_s']['sexist']))

        labels = {}
        for rew_id, pred in predictions.items():
            voting_score = defaultdict(lambda: 0)
            for label, confidence in pred:
                voting_score[label] += confidence

            label = max(voting_score.items(), key=lambda x: x[1])[0]
            confidence = max(voting_score.items(), key=lambda x: x[1])[1] / len(self.models)
            
            labels[rew_id] = {'sexist': label, 'confidence_s': {'sexist': confidence}}

        return labels, None

        