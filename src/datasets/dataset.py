import csv
from collections import defaultdict
from math import ceil

from torch.utils.data import Dataset
from transformers import AutoTokenizer
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

        self.tokenizer = AutoTokenizer.from_pretrained(configs.model.bert.name, use_fast=False)
        self.text_max_length = configs.model.bert.max_length


    def get_kth_fold_dataset(self, k):
        train_data = []
        test_data = []
        train_set, test_set =  self.k_splits[k]

        for i in train_set:
            train_data.append(self.data[i])
        for i in test_set:
            test_data.append(self.data[i])

        return EDOSDataset(f'train_{k}', self.configs, train_data), EDOSDataset(f'eval_{k}', self.configs, test_data)

    def duplicate_oversample(self):
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
    
    def back_translate_oversample(self):
        from translate import Translator
        from translate.exceptions import TranslationError
        from random import SystemRandom
        
        languages = ["es", "de", "fr", "ar", "te", "hi", "ja", "fa", "sq", "bg", "nl", "gu", "ig", "kk", "mt", "ps"]
        sr = SystemRandom()
        
        label_sexist_counter = defaultdict(int)

        for sample in self.data:
            label_sexist_counter[sample['label_sexist']] += 1

        max_count = max(label_sexist_counter.values())
        
        for label, count in label_sexist_counter.items():
            imbalance_count = max_count - count
            augmentation_multiplier = ceil(imbalance_count/count)
            english_translator = Translator(to_lang="en")
            
            if augmentation_multiplier: #* Will be 0 if label is the majority label
                for sample in self.data:
                    if sample['label_sexist'] == label:
                        
                        augmented_sample_list = []
                        for i in range(augmentation_multiplier):
                            random_translator = Translator(to_lang=sr.choice(languages))
                            try:
                                translated_text = random_translator.translate(sample['text'])
                                augmented_text = english_translator.translate(translated_text)
                            except TranslationError:
                                # Error Log
                                pass
                            augmented_sample = sample
                            augmented_sample['text'] = augmented_text
                            augmented_sample_list.append(augmented_sample)
                        
                        self.data += augmented_sample_list
                
    def oversample_the_dataset(self, strategy:str = None):
        if strategy == "duplicate":
            self.duplicate_oversample()
        
        elif strategy == "back_translate":
            self.back_translate_oversample()

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

        if self.configs.model.type == 'bert':
            encoding = self.tokenizer.encode_plus(
                item['text'],
                add_special_tokens=True,
                max_length=self.text_max_length,
                truncation= True,
                return_token_type_ids=False,
                padding = 'max_length',
                return_attention_mask=True,
                return_tensors='pt',
            )

            
            item['input_ids'] =  encoding['input_ids'].flatten()
            item['attention_mask'] =  encoding['attention_mask'].flatten()

        if self.configs.model.type == 'unifiedQA':
            sexism_definition = f'Sexism: any abuse or negative sentiment that is directed towards women based on their gender, or based on their gender combined with one or more other identity attributes.'
            question = f'Is the following sentence sexist?'
            options = ' '.join(['(A) sexist', '(B) not sexist'])
            evidence = item['text']
            unifiedQA_question = f'{question}\n{options}\n{evidence}\n{sexism_definition}'
            item['question'] = unifiedQA_question
        
        return item 


class TrainDataset(EDOSDataset):
    def __init__(self, configs):
        data = []
        with open(configs.train.file, newline='', encoding="utf8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
        
        super().__init__('train', configs, data)

class PredictionDataset(EDOSDataset):
    def __init__(self, configs):
        data = []
        with open(configs.predict.file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in tqdm(reader, desc='Loading eval dataset'):
                data.append(row)
        
        super().__init__('pred', configs, data)
