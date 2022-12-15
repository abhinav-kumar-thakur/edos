from src.models.bertweet_classifier import BertTweetClassifier

def get_classification_model(configs, device):
    model_name = configs.model.type

    if model_name == 'text':
        return BertTweetClassifier(configs, device)

    raise Exception('Invalid model name')
