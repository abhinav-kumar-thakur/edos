import os

from src.config_reader import read_json_configs
from src.models.utils import get_model

from .ensemble import Ensemble

class Individual(Ensemble):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.device = device
        self.configs = config
        
        self.model = None
        for model_config in self.configs.model.individual.models:
            c = read_json_configs(os.path.join('./configs', model_config['config']))
            self.model = get_model(c, model_config['path'], self.device)
        
    def forward(self, batch, train=False):
        return self.model(batch, train=False)
