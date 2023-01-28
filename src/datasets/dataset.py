import csv
from collections import defaultdict

from .base import BaseDataset

class EDOSDataset(BaseDataset):
    def __init__(self, name, data, k_fold):
        super().__init__(name, data, k_fold)
        
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
            
class TrainDataset(EDOSDataset):
    def __init__(self, dataset_name, dataset_path, tasks, k_fold):
        data = []
        dataset_path = dataset_path
        with open(dataset_path, newline='', encoding="utf8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if 'a' not in tasks and row['label_sexist'] == 'not sexist':
                    continue

                data.append(row)

        super().__init__(dataset_name, data, k_fold)

class AdditionalTrainDataset(EDOSDataset):
    def __init__(self, configs):
        data = []
        with open(configs.train.additional_files.sd,  newline='', encoding="utf-8-sig") as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            sexist = 0
            non_sexist = 0
            prefix = 'sd'
            for i, row in enumerate(rows):
                row['rewire_id'] = f'{prefix}_{i}'
                row['label_category'] = 'none'
                row['label_vector'] = 'none'
                if row['label_sexist']=='1':
                    sexist += 1
                    row['label_sexist'] = 'sexist'
                else:
                    non_sexist += 1
                    row['label_sexist'] = 'not sexist'

                data.append(row)

        super().__init__('train', configs, data, configs.train.k_fold)

class PredictionDataset(EDOSDataset):
    def __init__(self, dataset_path, k_fold=5):
        print(f'Using prediction dataset: {dataset_path}')
        data = []
        with open(dataset_path, newline='', encoding="utf8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)

        super().__init__('pred', data, k_fold)

class UnlabelledDataset(EDOSDataset):
    def __init__(self, name, unlabeled_file, k_fold):
        data = []
        count = 0
        with open(unlabeled_file, newline='', encoding="utf8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                row['rewire_id'] = f'{name}_{count}'
                row['label_sexist'] = 'unknown'
                row['label_category'] = 'unknown'
                row['label_vector'] = 'unknown'
                data.append(row)
                count += 1
        
        super().__init__('unlabeled', data, k_fold)
        

class DevDataset(EDOSDataset):
    def __init__(self, configs):
        data = defaultdict(dict)
        with open(configs.datasets.files.dev_a_text, newline='', encoding="utf8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data[row['rewire_id']] = {
                    'rewire_id': row['rewire_id'],
                    'text': row['text'],
                    'label_category': 'none',
                    'label_vector': 'none'
                } 

        with open(configs.datasets.files.dev_a_label, newline='', encoding="utf8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data[row['rewire_id']]['label_sexist'] = row['label']

        with open(configs.datasets.files.dev_b_label, newline='', encoding="utf8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data[row['rewire_id']]['label_category'] = row['label']

        with open(configs.datasets.files.dev_c_label, newline='', encoding="utf8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data[row['rewire_id']]['label_vector'] = row['label']
        
        super().__init__('dev', list(data.values()), configs.train.k_fold)

class MamiDataset(EDOSDataset):
    def __init__(self, configs):
        data = []
        prefix = 'mami'
        with open(configs.train.additional_files.mami, newline='', encoding="utf-8-sig") as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t')
            for row in reader:
                row['rewire_id'] = f'{prefix}_{row["file_name"]}'
                row['text'] = row['Text Transcription']
                row['label_category'] = 'none'
                row['label_vector'] = 'none'
                if row['misogynous'] == '1':
                    row['label_sexist'] = 'sexist'
                else:
                    row['label_sexist'] = 'not sexist'

                for key in ['file_name', 'Text Transcription', 'misogynous', 'shaming', 'stereotype', 'objectification', 'violence']:
                    del row[key]

                data.append(row)

        super().__init__('train', configs, data, configs.train.k_fold)