import os
import random
from pprint import pprint

from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.strategies.ensemble.utils import get_ensemble_model
from src.utils import get_args
from src.config_reader import read_json_configs, read_dict_configs
from src.logger import Logger
from src.datasets.dataset import TrainDataset, DevDataset, EDOSDataset

if __name__ == '__main__':
    args = get_args()
    configs = read_json_configs(os.path.join('./configs', args.config))

    logger = Logger(configs)
    state_configs_path = os.path.join(logger.dir, configs.logs.files.state)
    if os.path.exists(state_configs_path):
        state_configs = read_json_configs(state_configs_path)
    else:
        state_configs = read_dict_configs({'kth_fold': 0,
                                           'epoch': 0,
                                           'best_score': None,
                                           'epochs_without_improvement': 0,
                                           'kth_fold_metrics': []})

    torch.manual_seed(configs.seed)
    random.seed(configs.seed)

    # dataset = TrainDataset(configs)
    

    dataset = DevDataset(configs)
    from random import shuffle

    data = dataset.data
    shuffle(data)

    train_dataset = TrainDataset(configs, configs.train.files.train)
    eval_dataset = TrainDataset(configs, configs.train.files.eval)

    print('train:', train_dataset.summarize())
    print('eval:', eval_dataset.summarize())

    train_dataloader = DataLoader(train_dataset , batch_size=configs.train.train_batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=configs.train.eval_batch_size, shuffle=False, num_workers=0)

    ensemble_model = get_ensemble_model(configs, logger, args.device)
    ensemble_model.fit(train_dataloader)

    predictions = {}
    for batch in tqdm(eval_dataloader):
        pred, _ = ensemble_model(batch, train=False)
        for rew_id, pred in pred.items():
            predictions[rew_id] = pred['sexist']

    actual_labels = []
    prediction_labels = []
    for data in eval_dataset:
        rew_id = data['rewire_id']
        actual_labels.append(data['label_sexist'])
        prediction_labels.append(predictions[rew_id])

    print(classification_report(actual_labels, prediction_labels, output_dict=True))