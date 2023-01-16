from .voting import Voting
from .meta_classifier import MetaClassifier

from .random_forest_ensemble import RandomForestEnsembler

def get_ensemble_model(configs, logger, device):
    model_name = configs.model.type

    if model_name is None:
        raise Warning('No ensemble model specified')
    elif model_name == 'voting':
        model = Voting(configs, logger, device)
    elif model_name == 'meta-classifier':
        model = MetaClassifier(configs, logger, device)
    elif model_name == 'bagging_random_forest':
        model = RandomForestEnsembler(configs = configs, logger= logger, device= device)
    else:
        raise Exception('Invalid ensemble name')

    return model