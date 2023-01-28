from src.config_reader import read_json_configs
from src.datasets.dataset import UnlabelledDataset


configs_file = '/home/akthakur/edos/configs/B1-0-data-prep.json'
configs = read_json_configs(configs_file)

gab_unlabelled_dataset = UnlabelledDataset('gab', configs.datasets.files.gab_unlabeled, configs.train.k_fold)
reddit_unlabelled_dataset = UnlabelledDataset('reddit', configs.datasets.files.reddit_unlabeled, configs.train.k_fold)

merged_dataset = gab_unlabelled_dataset.merge(reddit_unlabelled_dataset)
print(merged_dataset.summarize())
merged_dataset.save('data/processed', 'unlabeled_combined')