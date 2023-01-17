import csv
import os
import random
from collections import defaultdict

from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import KFold
from tqdm import tqdm


class EDOSDataset(Dataset):
    def __init__(self, name, configs, data, k_fold):
        self.name = name
        self.data = data
        self.k_fold = k_fold
        self.kf = KFold(n_splits=self.k_fold, shuffle=True, random_state=42)
        self.k_splits = list(self.kf.split(self.data))
        self.configs = configs

    def get_kth_fold_dataset(self, k):
        train_data = []
        test_data = []
        train_set, test_set = self.k_splits[k]

        for i in train_set:
            train_data.append(self.data[i])
        for i in test_set:
            test_data.append(self.data[i])

        return EDOSDataset(f'train_{k}', self.configs, train_data, self.configs.train.k_fold), EDOSDataset(f'eval_{k}', self.configs, test_data, self.configs.train.k_fold)

    def oversample_the_dataset(self):
        # Create dict counter for each label
        label_sexist_counter = defaultdict(int)

        for sample in self.data:
            label_sexist_counter[sample['label_sexist']] += 1

        # Get the max count
        max_count = max(label_sexist_counter.values())

        # Create dict of lists for each label
        label_sexist_data = defaultdict(list)

        for sample in self.data:
            label_sexist_data[sample['label_sexist']].append(sample)

        # Duplicate the minority class samples to match the majority class
        for label, count in label_sexist_counter.items():
            if count < max_count:
                label_sexist_data[label] *= (max_count // count)

        # Concatenate the lists
        self.data = []
        for label, data in label_sexist_data.items():
            self.data += data

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        if self.configs.model.type == 'unifiedQA':
            sexism_definition = f'Sexism: any abuse or negative sentiment that is directed towards women based on their gender, or based on their gender combined with one or more other identity attributes.'
            question = f'Detect sexism and classify into category?'
            options = ' '.join([
                '(A) not sexist | none | none', 
                '(B) sexist | 1. threats, plans to harm and incitement | 1.1 threats of harm',
                '(C) sexist | 1. threats, plans to harm and incitement | 1.2 incitement and encouragement of harm',
                '(D) sexist | 2. derogation | 2.1 descriptive attacks',
                '(E) sexist | 2. derogation | 2.2 aggressive and emotive attacks',
                '(F) sexist | 2. derogation | 2.3 dehumanising attacks & overt sexual objectification',
                '(G) sexist | 3. animosity | 3.1 casual use of gendered slurs, profanities, and insults',
                '(H) sexist | 3. animosity | 3.2 immutable gender differences and gender stereotypes',
                '(I) sexist | 3. animosity | 3.3 backhanded gendered compliments',
                '(J) sexist | 3. animosity | 3.4 condescending explanations or unwelcome advice',
                '(K) sexist | 4. prejudiced discussions | 4.1 supporting mistreatment of individual women',
                '(L) sexist | 4. prejudiced discussions | 4.2 supporting systemic discrimination against women as a group'
            ])
            evidence = item['text']
            unifiedQA_question = f'{question}\n{options}\n{evidence}\n{sexism_definition}'
            item['question'] = unifiedQA_question
            item['answer'] = f'{item["label_sexist"]} | {item["label_category"]} | {item["label_vector"]}'

        return item

    def merge(self, dataset):
        merged_data = self.data.copy()
        for d in dataset.data:
            merged_data.append(d)

        return EDOSDataset('merged', dataset.configs, merged_data, dataset.configs.train.k_fold)

    def split(self, split_ratio: list):
        assert sum(split_ratio) == 1, 'split ratio must sum to 1'

        split_data = []
        split_count = len(split_ratio) 
        split_size = [int(len(self.data) * ratio) for ratio in split_ratio]
        random.shuffle(self.data)

        start = 0
        for i in range(split_count):
            end = start + split_size[i]
            split_data.append(EDOSDataset(f'split_{i}', self.configs, self.data[start:end], self.configs.train.k_fold))
            start = end

        return split_data

    def save(self):
        path = os.path.join(self.configs.data_dir, 'processed', self.configs.title + '-' + self.configs.task) 
        if not os.path.exists(path):
            os.makedirs(path)
        
        with open(os.path.join(path, f'{self.name}.csv'), 'w', newline='', encoding='utf8') as f:
            writer = csv.DictWriter(f, fieldnames=self.data[0].keys())
            writer.writeheader()
            for row in self.data:
                writer.writerow(row)
            
class TrainDataset(EDOSDataset):
    def __init__(self, configs, dataset_path=None, dataset_name='train'):
        data = []
        dataset_path = dataset_path if dataset_path else configs.datasets.files.train
        with open(dataset_path, newline='', encoding="utf8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if 'a' not in configs.train.task and row['label_sexist'] == 'not sexist':
                    continue

                data.append(row)

        super().__init__(dataset_name, configs, data, configs.train.k_fold)

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
    def __init__(self, configs):
        data = []
        with open(configs.datasets.files.test_a_text, newline='', encoding="utf8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in tqdm(reader, desc='Loading eval dataset'):
                data.append(row)

        super().__init__('pred', configs, data, configs.train.k_fold)



class UnlabelledDataset(EDOSDataset):
    def __init__(self, configs):
        data = []
        count = 0
        prefix = 'unlabelled'
        with open(configs.ssl.unlabelled_file, newline='', encoding="utf8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                row['rewire_id'] = f'{prefix}_{count}'
                data.append(row)
                count += 1
        
        super().__init__('unlabelled', configs, data, configs.ssl.k_fold)
        

class DevDataset(EDOSDataset):
    def __init__(self, configs):
        dev_a_text = {}
        dev_a_labels = {}
        dev_data_a = []
        with open(configs.datasets.files.dev_a_text, newline='', encoding="utf8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                dev_a_text[row['rewire_id']] = row['text']

        with open(configs.datasets.files.dev_a_label, newline='', encoding="utf8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                dev_a_labels[row['rewire_id']] = row['label']
        
        for key in dev_a_text:
            dev_data_a.append({
                'rewire_id': key,
                'text': dev_a_text[key],
                'label_sexist': dev_a_labels[key],
                'label_category': 'none',
                'label_vector': 'none'
            })
            
        super().__init__('dev', configs, dev_data_a, configs.train.k_fold)

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