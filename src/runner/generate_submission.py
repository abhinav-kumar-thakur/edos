import os

import torch
from tqdm import tqdm
from sklearn.metrics import classification_report

from src.config_reader import read_json_configs
from src.strategies.ensemble.utils import get_ensemble_model
from src.logger import Logger
from src.datasets.dataset import PredictionDataset, TrainDataset
from src.utils import get_args

if __name__ == '__main__':
    args = get_args()
    configs = read_json_configs(os.path.join('./configs', args.config))

    logger = Logger(configs)
    dataset = TrainDataset(configs, configs.submission.file) if configs.submission.dataset == 'dev' else PredictionDataset(configs)
    logger.log_text(configs.logs.files.event, f'Generating submission file for {configs.submission.file} dataset')
    
    model = get_ensemble_model(configs, logger, args.device)
    prediction_dataloader = torch.utils.data.DataLoader(dataset, batch_size=configs.train.eval_batch_size, shuffle=False, num_workers=0)
    
    predictions = {}
    for batch in tqdm(prediction_dataloader):
        pred, _ = model(batch, train=False)
        for rew_id, pred in pred.items():
            predictions[rew_id] = pred['sexist']

    actual_labels = []
    prediction_labels = []
    if configs.submission.dataset == 'dev':
        for data in dataset:
            rew_id = data['rewire_id']
            actual_labels.append(data['label_sexist'])
            prediction_labels.append(predictions[rew_id])

        print(classification_report(actual_labels, prediction_labels))
        logger.log_file(configs.logs.files.event, classification_report(actual_labels, prediction_labels, output_dict=True))
    
    output_file = os.path.join(logger.dir, configs.logs.files.submission)

    assert len(predictions) == len(dataset), f"Expected 4000 predictions, got {len(predictions)}"
    with open(output_file, 'w') as f:
        f.write('rewire_id,label_pred\n')
        for rew_id, pred in predictions.items():
            f.write(f'{rew_id},{pred}\n')

    print(f"Done generating submission file: {output_file}")
