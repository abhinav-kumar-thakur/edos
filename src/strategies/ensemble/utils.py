from .voting import Voting
from .meta_classifier import MetaClassifier
from .individual import Individual
from .weighted_voting import WeightedVoting
from .random_forest_ensemble import RandomForestEnsembler

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
    elif model_name == 'meta-classifier':
        model = MetaClassifier(configs, logger, device)
    elif model_name == 'bagging_random_forest':
        model = RandomForestEnsembler(configs = configs, logger= logger, device= device)
    else:
        raise Exception('Invalid ensemble name')

    return model