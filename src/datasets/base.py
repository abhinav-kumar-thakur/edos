import csv
import os
import random
from collections import defaultdict

from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from tqdm import tqdm


class BaseDataset(Dataset):
    def __init__(self, name, data, k_fold):
        self.name = name
        self.data = data
        self.k_fold = k_fold
        self.kf = KFold(n_splits=self.k_fold, shuffle=True, random_state=42)
        self.k_splits = list(self.kf.split(self.data))

    def get_kth_fold_dataset(self, k):
        train_data = []
        test_data = []
        train_set, test_set = self.k_splits[k]

        for i in train_set:
            train_data.append(self.data[i])
        for i in test_set:
            test_data.append(self.data[i])

        return BaseDataset(f'train_{k}', train_data, self.k_fold), BaseDataset(f'eval_{k}', test_data, self.k_fold)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def merge(self, dataset):
        merged_data = self.data.copy()
        for d in dataset.data:
            merged_data.append(d)

        return BaseDataset('merged', merged_data, self.k_fold)

    def split(self, split_ratio: list):
        assert sum(split_ratio) == 1, 'split ratio must sum to 1'

        split_data = []
        split_count = len(split_ratio) 
        split_size = [int(len(self.data) * ratio) for ratio in split_ratio]
        random.shuffle(self.data)

        start = 0
        for i in range(split_count):
            end = start + split_size[i]
            split_data.append(BaseDataset(f'split_{i}', self.data[start:end], self.k_fold))
            start = end

        return split_data

    def save(self, dir, name):
        with open(os.path.join(dir, f'{name}.csv'), 'w', newline='', encoding='utf8') as f:
            writer = csv.DictWriter(f, fieldnames=self.data[0].keys())
            writer.writeheader()
            for row in self.data:
                writer.writerow(row)

    def summarize(self):
        # Create dict counter for each label
        label_sexist_counter = defaultdict(int)
        label_category_counter = defaultdict(int)
        label_vector_counter = defaultdict(int)

        for sample in self.data:
            label_sexist_counter[sample['label_sexist']] += 1
            label_category_counter[sample['label_category']] += 1
            label_vector_counter[sample['label_vector']] += 1

        summary = {
            'name': self.name,
            'count': len(self.data),
            'label_sexist': dict(label_sexist_counter),
            'label_category': dict(label_category_counter),
            'label_vector': dict(label_vector_counter)
        }

        return summary
