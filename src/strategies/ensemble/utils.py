import os

import torch

from .voting import Voting
from .meta_classifier import MetaClassifier
from .individual import Individual
from .weighted_voting import WeightedVoting
from .random_forest_ensemble import RandomForestEnsembler
from .xgboost_ensembler import XGBoostEnsembler

def get_ensemble_model(configs, logger, device):
    model_name = configs.model.type

    if model_name is None:
        raise Warning('No ensemble model specified')
    elif model_name == 'individual':
        model = Individual(configs, device)
    elif model_name == 'voting':
        model = Voting(configs, device)
    elif model_name == 'weighted_voting':
        model = WeightedVoting(configs, device)
    elif model_name == 'meta_classifier':
        if configs.model.mode == 'train':
            model = MetaClassifier(configs, device)
        else:
            model = MetaClassifier(configs, device)
            model_path = os.path.join(configs.logs.dir, configs.title + '-' + configs.task, configs.logs.files.models, f'best_model_all_data.pt')
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f'Loaded model from {model_path}')
    elif model_name == 'bagging_random_forest':
        model = RandomForestEnsembler(configs = configs, logger= logger, device= device)
    elif model_name == 'xgboost_classifier':
        model = XGBoostEnsembler(configs = configs, logger= logger, device= device)
    else:
        raise Exception('Invalid ensemble name')

    return model