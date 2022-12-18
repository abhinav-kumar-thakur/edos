from src.models.bertweet_classifier import BertTweetClassifier
from src.models.unifiedQA import UnifiedQAClassifier

def get_classification_model(configs, device):
    model_name = configs.model.type

    if model_name == 'bert':
        return BertTweetClassifier(configs, device)

    if model_name == 'unifiedQA':
        return UnifiedQAClassifier(configs, device)

    raise Exception('Invalid model name')
