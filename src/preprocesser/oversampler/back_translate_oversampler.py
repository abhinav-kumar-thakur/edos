from translate import Translator
from translate.exceptions import TranslationError
from random import SystemRandom
from collections import defaultdict
from math import ceil
from tqdm import tqdm

LANGUAGES = ["es", "de", "fr", "ar", "te", "hi", "ja", "fa", "sq", "bg", "nl", "gu", "ig", "kk", "mt", "ps"]

def back_translate(english_translator:Translator, sample, augmentation_multiplier):
    sr = SystemRandom()
    augmented_samples = []
    
    for i in range(augmentation_multiplier):
        random_translator = Translator(to_lang=sr.choice(LANGUAGES))
        try:
            translated_text = random_translator.translate(sample['text'])
            augmented_text = english_translator.translate(translated_text)
        except Exception as e:
            print(i,sample['text'],e)
            raise Exception(e)
            pass

        augmented_sample = sample
        augmented_sample['text'] = augmented_text
        augmented_samples.append(augmented_sample)
    
    return augmented_samples
        

def back_translate_oversample(data:list):        
    label_sexist_counter = defaultdict(int)

    for sample in data:
        label_sexist_counter[sample['label_sexist']] += 1

    max_count = max(label_sexist_counter.values())

    for label, count in label_sexist_counter.items():
        imbalance_count = max_count - count
        augmentation_multiplier = ceil(imbalance_count/count)
        english_translator = Translator(to_lang="en")
        augemented_sample_list = []
        
        if augmentation_multiplier: #* Will be 0 if label is the majority label
            for sample in tqdm(data,desc="Augmentation Progress"):
                if sample['label_sexist'] == label:
                    augemented_samples= back_translate(english_translator, sample, augmentation_multiplier)
                    augemented_sample_list.extend(augemented_samples)

            data.extend(augemented_sample_list)    
    
    return data