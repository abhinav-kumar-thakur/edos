import csv

from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from tqdm import tqdm


class EDOSDataset(Dataset):
    def __init__(self, name, configs, data):
        self.name = name
        self.data = data
        self.k_fold = configs.train.k_fold
        self.kf = KFold(n_splits=self.k_fold, shuffle=True, random_state=42)
        self.k_splits = list(self.kf.split(self.data))
        self.configs = configs

    def get_kth_fold_dataset(self, k):
        train_data = []
        test_data = []
        train_set, test_set =  self.k_splits[k]
        
        for i in train_set:
            train_data.append(self.data[i])
        for i in test_set:
            test_data.append(self.data[i])

        return EDOSDataset(f'train_{k}', self.configs, train_data), EDOSDataset(f'eval_{k}', self.configs, test_data)

    def summarize(self):
        return f'{self.name} dataset has {len(self.data)} samples'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class TrainDataset(EDOSDataset):
    def __init__(self, configs):
        data = []
        with open(configs.train.file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in tqdm(reader, desc='Loading train dataset'):
                data.append(row)
        
        super().__init__('train', configs, data)