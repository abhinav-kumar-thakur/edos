import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import random

from src.config_reader import read_json_configs, read_dict_configs
from src.datasets.dataset import TrainDataset, DevDataset
from src.models.utils import get_classification_model_from_state
from src.optimizer.utils import get_optimizer
from src.logger import Logger
from src.utils import get_args
from src.trainer.edos_trainer import EDOSTrainer

if __name__ == '__main__':
    args = get_args()
    configs = read_json_configs(os.path.join('./configs', args.config))

    logger = Logger(configs)
    state_configs_path = os.path.join(logger.dir, configs.logs.files.state)
    if os.path.exists(state_configs_path):
        state_configs = read_json_configs(state_configs_path)
    else:
        state_configs = read_dict_configs({'kth_fold': 'all_data',
                                           'epoch': 0,
                                           'best_score': None,
                                           'epochs_without_improvement': 0,
                                           'kth_fold_metrics': []})

    torch.manual_seed(configs.seed)
    random.seed(configs.seed)

    train_dataset = TrainDataset(configs)
    eval_dataset = DevDataset(configs)
    train_dataloader = DataLoader(train_dataset, batch_size=configs.train.train_batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=configs.train.eval_batch_size, shuffle=False)
    
    model = get_classification_model_from_state(configs, state_configs, args.device)
    optimizer = get_optimizer(model, configs)    
    trainer = EDOSTrainer(configs, state_configs, model, train_dataloader, eval_dataloader, optimizer, args.device, logger)
    trainer.run()
    print("Finished training with all data")
