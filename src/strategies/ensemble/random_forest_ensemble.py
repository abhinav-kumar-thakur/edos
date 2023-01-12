from sklearn.ensemble import RandomForestClassifier
import random
from typing import List
from src.trainer.edos_trainer import EDOSTrainer
from torch.utils.data import DataLoader
from tqdm import tqdm
## Create a Random Forest Ensembler from multiple BERT model outputs
class RandomForestEnsembler:
    def __init__(self, n_estimators=100, max_depth=5, random_state=42,bootstrap=False,
                    use_frozen=True, classifiers:List[EDOSTrainer]=[]):
        self.random_state = random_state
        self.clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        self.bootstrap = bootstrap
        self.classifiers = classifiers
        self.use_frozen = use_frozen
    
    def fit(self, X, y):
        if self.use_frozen:
            data = DataLoader(X, batch_size=self.configs.train.eval_batch_size, shuffle=False)
            # 'pred_a', 'confidence_a', 'uncertainity_a'
            classifier_outputs = [[pred[3:6] for pred in classifier.eval(data)[1][1:]] for classifier in tqdm(self.classifiers)]
            classifier_outputs = [sum(op,[]) for op in zip(*classifier_outputs)]
            self.clf.fit(classifier_outputs, y)
    
    def predict(self, X):
        if self.use_frozen:
            data = DataLoader(X, batch_size=self.configs.train.eval_batch_size, shuffle=False)
            classifier_outputs = [[pred[3:6] for pred in classifier.eval(data)[1][1:]] for classifier in tqdm(self.classifiers)]
            classifier_outputs = [sum(op,[]) for op in zip(*classifier_outputs)]
            return self.clf.predict(X)
        return
    
    def bootstrap_data(X, n:int, bootstrap_frac:float=0.75):
        return [random.sample(X, len(X)*bootstrap_frac) for _ in range(n)]

# Path: src\strategies\ensemble\ensemble.py
