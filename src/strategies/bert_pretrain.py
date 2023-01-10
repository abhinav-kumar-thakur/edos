import torch
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, \
    Trainer, TrainingArguments
from torch.utils.data import Dataset
from argparse import ArgumentParser
from time import time
import csv
from typing import Dict
import logging
import json
from tqdm import tqdm
logging.basicConfig(filename='logs/bert_pretrain_log.log', format='%(asctime)s %(message)s', encoding='utf-8', level=logging.DEBUG, filemode='w')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PreTrainDataset(Dataset):

    def __init__(self, tokenizer, data, block_size) -> None:
        super().__init__()
        logger.info("Initializing Dataset")
        temp_t = time()
        batch_encoding = tokenizer(data, add_special_tokens=True, truncation=True, max_length=block_size)
        logger.info(f"Tokenized in {time()-temp_t} seconds")
        self.samples = []
        for e in tqdm(batch_encoding["input_ids"]):
            new_ip = {"input_ids": torch.tensor(e,device=DEVICE)}
            self.samples.append(new_ip)
        logger.info(f"Custom Dataset Initialized with device {DEVICE}")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.samples[i]

def pretrain(data:list):
    logger.info(f"GPU {'' if torch.cuda.is_available() else 'not'} available")
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    dataset = PreTrainDataset(tokenizer=tokenizer, data=data, block_size=config['tokenizer_batch_size'])

    bert_config = BertConfig(vocab_size=50000, hidden_size=768, num_hidden_layers=6, 
        num_attention_heads=12, max_position_embeddings=512
    )
    model = BertForMaskedLM(config=bert_config)
    model.to(DEVICE)

    training_args = TrainingArguments(
        output_dir = config['training_args']['output_dir'],
        overwrite_output_dir = config['training_args']['overwrite_output_dir'],
        num_train_epochs = config['training_args']['num_train_epochs'],
        per_device_train_batch_size = config['training_args']['per_device_train_batch_size'],
        save_steps = config['training_args']['save_steps'],
        save_total_limit = config['training_args']['save_total_limit'],
        logging_dir='logs/bert_pretrain_log_training.log'
    )

    logger.info(f"Training Arguments Initialized: {training_args.to_dict()}")   

    trainer = Trainer(model=model, args=training_args, data_collator=data_collator, train_dataset=dataset)

    t = time()
    trainer.train()
    logger.info(f"Training Complete in {time()-t} seconds")
    trainer.save_model(config['training_args']['output_dir'])

def get_data(file_path:str):
    data = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row['text']) if 'text' in row else data.append(row[row.keys()[0]])
    csvfile.close()
    return data

def get_all_data():
    complete_data = []
    for file_path in config['file_paths']:
        data = get_data(file_path)
        logger.info(f'Data Input File Path:{file_path}\tData Count:{len(data)}')
        complete_data.extend(data)
    logger.info(f"Complete Pretraining Data Length {len(complete_data)}")
    
    if config['save_combined_data']:
        combined_path = 'data/raw/'+'_'.join([f.split('/')[-1].split('.')[0] for f in config['file_paths']])+'_combined.csv'
        with open(combined_path,'w',newline='',encoding='utf-8') as combined_csv:
            writer = csv.writer(combined_csv)
            writer.writerow(['text'])
            writer.writerows([list(row) for row in data])
        logger.info(f"Combined Data Saved at {combined_path}")        
    
    return complete_data

def read_configs(config_file):
    with open(config_file, "r") as f:
        config = json.load(f) 
    return config  

if __name__ == "__main__":
    arg_parser = ArgumentParser(prog="BERT Pretrainer with MLM")
    # arg_parser.add_argument('-fs','--files', nargs='+', required=True, help='Files for %(prog)s')
    arg_parser.add_argument('-c','--config',required=True,default="configs/bert_pretrain.json")
    args = arg_parser.parse_args()

    config = read_configs(args.config)
    
    assert len(config['file_paths']) > 0, "No File Paths Provided"

    logger = logging.getLogger()

    pretrain_data = get_all_data()

    pretrain(pretrain_data)