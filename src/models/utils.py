import os

import torch

from src.models.bert import BertClassifier
from src.models.bert_focal_loss import BertClassifier_fl
from src.models.unifiedQA import UnifiedQAClassifier


def get_classification_model_from_state(configs, state_configs, device):
    print('Loading model...')
    model_name = configs.model.type

    if model_name == 'bert':
        model = BertClassifier(configs, device)
    elif model_name == 'bert_fl':
        model = BertClassifier_fl(configs, device)
    elif model_name == 'unifiedQA':
        model = UnifiedQAClassifier(configs, device)
    else:
        raise Exception('Invalid model name')

    if state_configs.epoch > 0:
        print(f'Loading saved model for {state_configs.kth_fold} fold and {state_configs.epoch} epoch')
        saved_model_path = os.path.join(configs.logs.dir, configs.title + '-' + configs.task, configs.logs.files.models, f'saved_model_state.pt')
        model.load_state_dict(torch.load(saved_model_path))

    return model


def save_model(model, configs):
    saved_model_path = os.path.join(configs.logs.dir, configs.title + '-' + configs.task, configs.logs.files.models, f'saved_model_state.pt')
    torch.save(model.state_dict(), saved_model_path)

def get_model(configs, filepath, device):
    model_name = configs.model.type
    if model_name == 'bert':
        model = BertClassifier(configs, device)
    elif model_name == 'bert_fl':
        model = BertClassifier_fl(configs, device)
    elif model_name == 'unifiedQA':
        model = UnifiedQAClassifier(configs, device)
    else:
        raise Exception('Invalid model name')

    model.load_state_dict(torch.load(filepath, map_location=device))
    return model
