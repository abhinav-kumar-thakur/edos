import os
import random
from argparse import ArgumentParser
import csv

from oversampler.duplicate_oversampler import duplicate_oversample
from oversampler.back_translate_oversampler import back_translate_oversample
from oversampler.paraphrase_oversampler import paraphraser
from src.config_reader import read_json_configs
from src.datasets.dataset import TrainDataset

def oversample(strategy:str, data:list):
    print("Strategy",strategy)
    if strategy == "duplicate":
        data = duplicate_oversample(data)
    
    elif strategy == "back_translate":
        data = back_translate_oversample(data)
    
    elif strategy == "paraphrase":
        data = paraphraser(data)

    else:
        raise ValueError("Incorrect strategy type input")

    return data
    
def save_oversampled_data(data:TrainDataset):
    if not os.path.exists(data.configs.datasets.preprocess.oversampled_data_dir):
        os.makedirs(data.configs.datasets.preprocess.oversampled_data_dir,exist_ok=True)
        
        save_path = os.path.join(data.configs.datasets.preprocess.oversampled_data_dir,data.configs.datasets.preprocess.oversampled_data_file)
        with open(save_path,'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(data.data)

if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--config', default='dev.json', required=True, help='Config file from ./configs')
    args = arg_parser.parse_args()

    configs = read_json_configs(os.path.join('./configs', args.config))

    random.seed(configs.seed)

    data = TrainDataset(configs)

    if "preprocess" in configs.configs['datasets']:
        if "oversampling_strategy" in configs.configs['datasets']['preprocess']:
            if configs.datasets.preprocess.oversampling_strategy: 
                augmented_data = oversample(configs.datasets.preprocess.oversampling_strategy,data)
                save_oversampled_data(augmented_data)
            
            else: print("No Oversampling Performed")
        else: raise ValueError("'oversampling_strategy' key missing in config")
    else: raise ValueError("'preprocess' key missing in config")