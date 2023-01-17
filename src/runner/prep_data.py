import os

from tqdm import tqdm

from src.config_reader import read_json_configs
from src.logger import Logger
from src.datasets.dataset import DevDataset, TrainDataset
from src.utils import get_args

if __name__ == '__main__':
    args = get_args()

    configs = read_json_configs(os.path.join('./configs', args.config))

    logger = Logger(configs)

    train_dataset = TrainDataset(configs)
    eval_dataset = DevDataset(configs)
    merged_dataset = eval_dataset.merge(train_dataset)

    print(f"train dataset: ", train_dataset.summarize())
    print(f"eval dataset: ", eval_dataset.summarize())
    print(f"merged dataset: ", merged_dataset.summarize())
    
    splits = [0.7, 0.15, 0.15]
    dataset_splits = merged_dataset.split(splits)

    for dataset in dataset_splits:
        dataset.save()
        print(f"split datatset: ", dataset.summarize())
