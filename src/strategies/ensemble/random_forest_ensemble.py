from sklearn.ensemble import RandomForestClassifier
import random
from typing import List
from ...trainer.edos_trainer import EDOSTrainer
from ...models.utils import get_model
from ...logger import Logger
from .ensemble import Ensemble
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
import os
import pickle

class RandomForestEnsembler(Ensemble):
    def __init__(self, configs, logger:Logger, device='cpu',load_path=None):
        super().__init__()

        self.device = device
        self.logger = logger
        self.configs = configs
        self.model_dir = os.path.join(self.logger.dir, self.configs.logs.files.models)
        self.log_file = self.configs.logs.files.ensemble
        
        self.classifiers:List[EDOSTrainer] = []
        for file in os.listdir(self.model_dir):
            if 'best_model' in file:
                model = get_model(self.configs, os.path.join(self.model_dir, file), self.device)
                self.logger.log(f"Clasifier loaded from {os.path.join(self.model_dir, file)}")
                self.classifiers.append(model)
        
        self.rf_parameters = self.configs.model.ensemble.parameters.configs
        self.random_state = self.rf_parameters['random_state']
        if load_path is None:
            self.clf = RandomForestClassifier()
            self.clf.set_params(**self.rf_parameters)
            self.logger.log_file(self.log_file, f"Random Forest Ensembler Loaded with parameters {self.clf.get_params()}")
        else:
            self.clf:RandomForestClassifier = pickle.load(open(load_path, 'rb'))
            self.logger.log_file(self.log_file, f"Random Forest Ensembler Loaded from {load_path} with parameters {self.clf.get_params()}")

        self.use_frozen = self.configs.model.ensemble.use_frozen
        self.bootstrap = self.configs.model.ensemble.bootstrap_data
    
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
            pred, loss = model(batch, train=train)
            for i, rewire_id in enumerate(batch['rewire_id']):
                predictions[rewire_id].append((
                    pred[rewire_id]['sexist'] if 'a' in self.configs.train.task else '-',
                    pred[rewire_id]['confidence']['sexist'] if 'a' in self.configs.train.task else '-',
                    pred[rewire_id]['uncertainity']['sexist'] if 'a' in self.configs.train.task else '-'))
                if train: y.append(batch['label_sexist'][i])
        
        rf_input = [sum(cl_ops,()) for cl_ops in tqdm(predictions.values(), desc='Reformatting Batch')]
        if train: self.clf.fit(rf_input, y)
        else: return self.clf.predict(rf_input)
    
    def bootstrap_data(self, X):
        n = self.bootstrap['n']
        bootstrap_frac = self.bootstrap['bootstrap_frac']
        return [random.sample(X, len(X)*bootstrap_frac) for _ in range(n)]

# Path: src\strategies\ensemble\ensemble.py
