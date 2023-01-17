from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from src.models.bert import BertClassifier
from src.models.bert_focal_loss import BertClassifier_fl
from src.models.unifiedQA import UnifiedQAClassifier
from src.config_reader import read_json_configs
import random
from typing import List
from ...trainer.edos_trainer import EDOSTrainer
from ...logger import Logger
from .ensemble import Ensemble
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
import os
import pickle
import torch as t

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

    model.load_state_dict(t.load(filepath, map_location=device))
    return model


class RandomForestEnsembler(Ensemble):
    def __init__(self, configs, logger:Logger, device='cpu',load_path=None):
        super().__init__()

        self.device = device
        self.logger = logger
        self.configs = configs
        # self.model_dirs = [os.path.join(configs.logs.dir,classifier_dir) for classifier_dir in configs.model.ensemble.classifier_dirs]
        # self.model_types = configs.model.ensemble.classifier_types
        # self.model_dir = os.path.join(self.logger.dir, self.configs.logs.files.models)
        # self.logger.log(str(self.model_dirs))
        self.log_file = self.configs.logs.files.ensemble
        
        self.classifiers:List[EDOSTrainer] = []
        # for classifier_dir, classifier_type in zip(self.model_dirs, self.model_types):
        #     models_dir = os.path.join(classifier_dir,'models')
        #     for file in os.listdir(models_dir):
        #         if 'best_model' in file:
        #             model_dir = os.path.join(models_dir, file)
        #             model = get_model(self.configs, model_dir, self.device)
        #             self.logger.log(f"Best Classifier Model loaded from {models_dir}/{file}")
        #             self.classifiers.append(model)

        for model_config in self.configs.model.bagging_random_forest.models:
            c = read_json_configs(os.path.join(
                './configs', model_config['config']))
            model = get_model(c, model_config['path'], 'cuda')
            self.classifiers.append(model)
        
        self.rf_parameters = self.configs.model.bagging_random_forest.parameters.configs
        self.random_state = self.rf_parameters['random_state']
        if load_path is None:
            self.clf = RandomForestClassifier()
            self.clf.set_params(**self.rf_parameters)
            self.logger.log_file(self.log_file, f"Random Forest Ensembler Loaded with parameters {self.clf.get_params()}")
        else:
            self.clf:RandomForestClassifier = pickle.load(open(load_path, 'rb'))
            self.logger.log_file(self.log_file, f"Random Forest Ensembler Loaded from {load_path} with parameters {self.clf.get_params()}")

        self.use_frozen = self.configs.model.bagging_random_forest.use_frozen
        self.bootstrap = self.configs.model.bagging_random_forest.bootstrap_data
        self.encoding_map_task_a = {}
    
    def fit(self, dataloader:DataLoader):       
        for batch in tqdm(dataloader, desc='Fitting Random Forest'):
            self.forward(batch, train=True)
        self.logger.log_text(self.log_file,"Random Forest Ensembler Trained")
        self.logger.log("Random Forest Ensembler Trained")
        save_path = os.path.join(self.configs.logs.dir, self.configs.title + '-' + self.configs.task, self.configs.logs.files.models, f'random_forest_ensembler.pickle')
        pickle.dump(self.clf, open(save_path, 'wb'))
        self.logger.log(f"Random Forest Ensembler Saved at {save_path}")
        self.logger.log_text(self.log_file, f"Random Forest Ensembler Saved at {save_path}")
    
    def forward(self, batch, train=False):
        predictions = defaultdict(list)
        y = []
        for model in self.classifiers:
            model.eval()
            pred, loss = model(batch, train= (not self.use_frozen))
            for i, rewire_id in enumerate(batch['rewire_id']):
                pred_label = pred[rewire_id]['sexist']
                if pred_label not in self.encoding_map_task_a:
                    self.encoding_map_task_a[pred_label] = len(self.encoding_map_task_a)+1
                encoded_pred_label = self.encoding_map_task_a[pred_label]
                predictions[rewire_id].append((
                    encoded_pred_label if 'a' in self.configs.train.task else '-',
                    pred[rewire_id]['confidence']['sexist'] if 'a' in self.configs.train.task else '-',
                    pred[rewire_id]['uncertainity']['sexist'] if 'a' in self.configs.train.task else '-'))
                if train: 
                    label = batch['label_sexist'][i]
                    if label not in self.encoding_map_task_a:
                        self.encoding_map_task_a[label] = len(self.encoding_map_task_a)+1
                    encoded_label = self.encoding_map_task_a[label]
                    y.append(encoded_label)
        
        rf_input = [sum(cl_ops,()) for cl_ops in tqdm(predictions.values(), desc='Reformatting Batch', leave=False)]
        if train: self.clf.fit(rf_input, y)
        else: return self.clf.predict(rf_input)
    
    def bootstrap_data(self, X):
        n = self.bootstrap['n']
        bootstrap_frac = self.bootstrap['bootstrap_frac']
        return [random.sample(X, len(X)*bootstrap_frac) for _ in range(n)]

# Path: src\strategies\ensemble\ensemble.py
