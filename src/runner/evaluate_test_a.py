import csv
from pprint import pprint

from sklearn.metrics import classification_report

from src.datasets.base import EDOSDataset, PredictionDataset

test_a_entries_file_path = 'data/raw/test_task_a_entries.csv'
prediction_file_path = 'logs/A1-2-meta-classifier/a2z-task-a-final.csv'
true_label_file_path = 'data/raw/test_task_b_entries.csv'



prediction_dict = {}
with open(prediction_file_path, 'r') as prediction_file:
    prediction_reader = csv.DictReader(prediction_file, delimiter=',')
    for row in prediction_reader:
        prediction_dict[row['rewire_id']] = {
            'prediction': row['label_pred'],
            'actual': 'not sexist',
        }

test_a_pred_dataset = PredictionDataset(None, './data/raw/test_task_a_entries.csv')
for d in test_a_pred_dataset:
    prediction_dict[d['rewire_id']]['text'] = d['text']

with open(true_label_file_path, 'r') as true_label_file:
    true_label_reader = csv.DictReader(true_label_file, delimiter=',')
    for row in true_label_reader:
        prediction_dict[row['rewire_id']]['actual'] = 'sexist'

data = []
actual = []
prediction = []
for key in prediction_dict:
    prediction.append(prediction_dict[key]['prediction'])
    actual.append(prediction_dict[key]['actual'])
    data.append({
        'rewire_id': key,
        'text': prediction_dict[key]['text'],
        'label_sexist': prediction_dict[key]['prediction'],
        'label_category': 'none',
        'label_vector': 'none',
    })


pprint(classification_report(actual, prediction, output_dict=True))

edos_dataset = EDOSDataset('test_a', None, data, 2)
edos_dataset.save('data/processed/test_a.csv')

"""
{'accuracy': 0.8915,
 'macro avg': {'f1-score': 0.8478772390349809,
               'precision': 0.8594484587415179,
               'recall': 0.8379674049879214,
               'support': 4000},
 'not sexist': {'f1-score': 0.9293389775317487,
                'precision': 0.9170951156812339,
                'recall': 0.9419141914191419,
                'support': 3030},
 'sexist': {'f1-score': 0.7664155005382132,
            'precision': 0.8018018018018018,
            'recall': 0.734020618556701,
            'support': 970},
 'weighted avg': {'f1-score': 0.8898300343608164,
                  'precision': 0.8891364870654717,
                  'recall': 0.8915,
                  'support': 4000}}
"""