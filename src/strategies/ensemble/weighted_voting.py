import os
import itertools
from collections import defaultdict

from src.config_reader import read_json_configs
from src.models.utils import get_model
from .ensemble import Ensemble


class WeightedVoting(Ensemble):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.device = device
        self.models = []
        self.metrics = []
        self.configs = config

        self.model = None
        for model_config in self.configs.model.weighted_voting.models:
            c = read_json_configs(os.path.join('./configs', model_config['config']))
            m = read_json_configs(model_config['metrics']).configs
            model = get_model(c, model_config['path'], self.device)
            self.models.append(model)
            self.metrics.append(m)

    def forward(self, batch, train=False):
        predictions = defaultdict(list)
        for i, model in enumerate(self.models):
            model.eval()
            pred, loss = model(batch, train=False)
            metrics = self.metrics[i]
            for rewire_id in batch['rewire_id']:
                predictions[rewire_id].append({
                    'label': pred[rewire_id]['sexist'],
                    'weight': metrics['eval_metric']['a']['macro avg']['f1-score']
                })

        labels = {}
        for rew_id, pred in predictions.items():
            label_dict = defaultdict(float)
            weight_sum = 0
            for p in pred:
                label_dict[p['label']] += p['weight']
                weight_sum += p['weight']

            label = max(label_dict, key=label_dict.get)
            labels[rew_id] = {
                'sexist': label,
                'confidence_s': {'sexist': label_dict[label]/weight_sum}
            }

        return labels, None
