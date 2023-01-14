from .voting import Voting
from .meta_classifier import MetaClassifier


def get_ensemble_model(configs, logger, device):
    model_name = configs.model.ensemble

    if model_name == 'voting':
        model = Voting(configs, logger, device)
    elif model_name == 'meta-classifier':
        model = MetaClassifier(configs, logger, device)
    else:
        raise Exception('Invalid ensemble name')

    return model