from .voting import Voting

def get_ensemble_model(configs, logger, device):
    model_name = configs.model.ensemble

    if model_name == 'voting':
        model = Voting(configs, logger, device)
    else:
        raise Exception('Invalid model name')

    return model