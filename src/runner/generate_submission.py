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
    dataset = TrainDataset('train', configs.submission.file, configs.train.task, configs.train.k_fold) if configs.submission.dataset == 'dev' else PredictionDataset(configs.submission.file, configs.train.k_fold)
    logger.log_text(configs.logs.files.event, f'Generating submission file for {configs.submission.file} dataset')
    
    model = get_ensemble_model(configs, logger, args.device)
    prediction_dataloader = torch.utils.data.DataLoader(dataset, batch_size=configs.train.eval_batch_size, shuffle=False, num_workers=0)
    
    predictions_a, predictions_b, predictions_c = {}, {}, {}
    for batch in tqdm(prediction_dataloader):
        pred, _ = model(batch, train=False)
        for rew_id, pred in pred.items():
            predictions_a[rew_id] = pred['sexist']
            predictions_b[rew_id] = pred['category']
            predictions_c[rew_id] = pred['vector']

    actual_labels_a, actual_labels_b, actual_labels_c = [], [], []
    prediction_labels_a, prediction_labels_b, prediction_labels_c = [], [], []
    if configs.submission.dataset == 'dev':
        for data in dataset:
            rew_id = data['rewire_id']
            if 'a' in configs.train.task:
                actual_labels_a.append(data['label_sexist'])
                prediction_labels_a.append(predictions_a[rew_id])
            if 'b' in configs.train.task:   
                actual_labels_b.append(data['label_category'])
                prediction_labels_b.append(predictions_b[rew_id])
            if 'c' in configs.train.task:
                actual_labels_c.append(data['label_vector'])
                prediction_labels_c.append(predictions_c[rew_id])
        classification_reports = {}
        if 'a' in configs.train.task:
            print('*'*5, 'Task A', '*'*5)
            report_a = classification_report(actual_labels_a, prediction_labels_b)
            classification_reports['a'] = report_a
            print(report_a)
            print('*'*18)
        if 'b' in configs.train.task:
            print('*'*5, 'Task B', '*'*5)
            report_b = classification_report(actual_labels_b, prediction_labels_b)
            classification_reports['b'] = report_b
            print(report_b)
            print('*'*18)
        if 'c' in configs.train.task:
            print('*'*5, 'Task C', '*'*5)
            report_c = classification_report(actual_labels_c, prediction_labels_c)
            classification_reports['c'] = report_c
            print(report_c)
            print('*'*18)  

        logger.log_file(configs.logs.files.event, classification_reports)
    
    if 'a' in configs.train.task:
        output_file_a = os.path.join(logger.dir, 'a-'+configs.logs.files.submission)
        with open(output_file_a, 'w') as f:
            f.write('rewire_id,label_pred\n')
            for rew_id, pred in predictions_a.items():
                pred_string = f'"{pred}"' if ',' in pred else pred
                f.write(f'{rew_id},{pred_string}\n')
        print(f"Done generating submission file for Task A: {output_file_a}")
    if 'b' in configs.train.task:
        output_file_b = os.path.join(logger.dir, 'b-'+configs.logs.files.submission)
        with open(output_file_b, 'w') as f:
            f.write('rewire_id,label_pred\n')
            for rew_id, pred in predictions_b.items():
                pred_string = f'"{pred}"' if ',' in pred else pred
                f.write(f'{rew_id},{pred_string}\n')
        print(f"Done generating submission file for Task B: {output_file_b}")
    if 'c' in configs.train.task:
        output_file_c = os.path.join(logger.dir, 'c-'+configs.logs.files.submission)
        with open(output_file_c, 'w') as f:
            f.write('rewire_id,label_pred\n')
            for rew_id, pred in predictions_c.items():
                pred_string = f'"{pred}"' if ',' in pred else pred
                f.write(f'{rew_id},{pred_string}\n')
        print(f"Done generating submission file for Task C: {output_file_c}")

    # assert len(predictions) == len(dataset), f"Expected 4000 predictions, got {len(predictions)}"
