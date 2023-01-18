import os
import itertools
from collections import defaultdict, Counter

from src.config_reader import read_json_configs
from src.models.utils import get_model
from .ensemble import Ensemble

class Voting(Ensemble):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.device = device
        self.models = []
        self.configs = config

        self.model = None
        for model_config in self.configs.model.voting.models:
            c = read_json_configs(os.path.join('./configs', model_config['config']))
            model = get_model(c, model_config['path'], self.device)
            self.models.append(model)
             
    def forward(self, batch, train=False):
        predictions = defaultdict(list)
        for model in self.models:
            model.eval()
            pred, loss = model(batch, train=False)
            for rewire_id in batch['rewire_id']:
                predictions[rewire_id].append(pred[rewire_id]['sexist'])

        labels = {}
        for rew_id, pred in predictions.items():
            p = Counter(pred).most_common(1)[0]
            labels[rew_id] = {
                'sexist': p[0], 
                'confidence_s': {'sexist': p[1]/len(pred)}
            }

        return labels, None
