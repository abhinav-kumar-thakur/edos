import json

from src.config_reader import read_json_configs
from src.datasets.dataset import TrainDataset, DevDataset


configs_file = '/home/akthakur/edos/configs/B1-0-data-prep.json'
configs = read_json_configs(configs_file)

train_dataset = TrainDataset('train', '/home/akthakur/edos/data/raw/train_all_tasks.csv', ['a'], 5)
eval_dataset = DevDataset(configs)
test_dataset = TrainDataset('test', '/home/akthakur/edos/data/processed/test_a.csv', ['a'], 5)


merged_dataset = eval_dataset.merge(train_dataset)
merged_dataset = merged_dataset.merge(test_dataset)

print(merged_dataset.summarize())
merged_dataset.save('data/processed', 'merged_a')