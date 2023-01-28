import os
import json

from tqdm import tqdm

from src.config_reader import read_json_configs
from src.datasets.dataset import PredictionDataset, TrainDataset, EDOSDataset
from src.utils import get_args

if __name__ == '__main__':
    args = get_args()
    configs = read_json_configs(os.path.join('./configs', args.config))

    dataset = TrainDataset(configs, configs.submission.file) if configs.submission.dataset == 'dev' else PredictionDataset(configs.submission.file, 5)
    predictions = json.load(open('a_pred.json'))


    sexist_data = []
    for data in tqdm(dataset):
        if predictions[data['rewire_id']]['label'] == 'sexist':
            data['label_sexist'] = 'sexist'

            if predictions[data['rewire_id']]['confidence'] >= 0.6:
                data['label_sexist'] = 'sexist'
                sexist_data.append(data)


    new_dataset = EDOSDataset('sexist_unlabeled', sexist_data, 5)
    new_dataset.save('data/processed', 'sexist_unlabeled')

    print(new_dataset.summarize())
    print(len(sexist_data))

        



