import torch as t
from transformers import AutoModel

from src.lossFunctions.focal_loss import FocalLoss

class BertClassifier_fl(t.nn.Module):
    def __init__(self, configs, device='cpu') -> None:
        super().__init__()

        self.device = device
        self.configs = configs

        task_a_loss_weights = t.FloatTensor(self.configs.model.bert.heads.a.loss_weights).to(device)
        self.loss_a = FocalLoss(alpha=task_a_loss_weights, gamma=2)

        self.label2idx_a = self.get_label_index_a()
        self.idx2label_a = {v: k for k, v in self.label2idx_a.items()}

        task_b_loss_weights = t.FloatTensor(self.configs.model.bert.heads.b.loss_weights).to(device)
        self.loss_b = FocalLoss(alpha=task_b_loss_weights)
        self.label2idx_b = self.get_label_index_b()
        self.idx2label_b = {v: k for k, v in self.label2idx_b.items()}

        task_c_loss_weights = t.FloatTensor(self.configs.model.bert.heads.c.loss_weights).to(device)
        self.loss_c = FocalLoss(alpha=task_c_loss_weights)
        self.label2idx_c = self.get_label_index_c()
        self.idx2label_c = {v: k for k, v in self.label2idx_c.items()}

        self.bert = AutoModel.from_pretrained(configs.model.bert.name).to(device)
        self.head_a = t.nn.Linear(configs.model.bert.dimensions, len(self.label2idx_a)).to(device)
        self.head_b = t.nn.Linear(configs.model.bert.dimensions, len(self.label2idx_b)).to(device)
        self.head_c = t.nn.Linear(configs.model.bert.dimensions, len(self.label2idx_c)).to(device)

        if self.configs.model.bert.freeze_lower_layers != 0:
            freeze_lower_layers = self.configs.model.bert.freeze_lower_layers
            
            for layer in self.bert.encoder.layer[:freeze_lower_layers]:
                for param in layer.parameters():
                    param.requires_grad = False


    def get_label_index_a(self):
        return {x: y['id'] for x, y in self.configs.datasets.labels.configs.items()}

    def get_label_index_b(self):
        label2idx, i = {}, 0
        for label_value in self.configs.datasets.labels.configs.values():
            for category in label_value['categories']:
                label2idx[category] = i
                i += 1
        return label2idx

    def get_label_index_c(self):
        label2idx, i = {}, 0
        for label_value in self.configs.datasets.labels.configs.values():
            for category in label_value['categories'].values():
                for vector in category['vectors']:
                    label2idx[vector] = i
                    i += 1
        return label2idx

    def forward(self, batch, train=True):
        input_ids_a = batch['input_ids'].to(self.device)
        attention_mask_a = batch['attention_mask'].to(self.device)

        _, pooled_output = self.bert(input_ids=input_ids_a, attention_mask=attention_mask_a, return_dict=False)
        x = t.relu(pooled_output)
        pred_a = self.head_a(x)
        pred_b = self.head_b(x)
        pred_c = self.head_c(x)
        
        loss = 0
        if train:
            if 'a' in self.configs.train.task:
                actual_a = t.tensor([self.label2idx_a[l] for l in batch['label_sexist']]).to(self.device)
                loss_a = self.loss_a(pred_a, actual_a)
                loss += loss_a
            
            if 'b' in self.configs.train.task:
                actual_b = t.tensor([self.label2idx_b[l] for l in batch['label_category']]).to(self.device)
                loss_b = self.loss_b(pred_b, actual_b)
                loss += loss_b

            if 'c' in self.configs.train.task:
                actual_c = t.tensor([self.label2idx_c[l] for l in batch['label_vector']]).to(self.device)
                loss_c = self.loss_c(pred_c, actual_c)
                loss += loss_c


        pred_a_ids = t.argmax(pred_a, dim=1)
        pred_b_ids = t.argmax(pred_b, dim=1)
        pred_c_ids = t.argmax(pred_c, dim=1)

        labels = {}
        for i in range(len(pred_a_ids)):
            labels[batch['rewire_id'][i]] = {
                'sexist': self.idx2label_a[pred_a_ids[i].item()],
                'category': self.idx2label_b[pred_b_ids[i].item()],
                'vector': self.idx2label_c[pred_c_ids[i].item()]
            }

        return labels, loss

    def get_trainable_parameters(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in self.bert.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr': self.configs.train.optimizer.lr * 0.1},

            {'params': [p for n, p in self.bert.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': self.configs.train.optimizer.lr * 0.1},

            {'params': [p for n, p in self.head_a.named_parameters()], 'weight_decay': 0.01, 'lr': self.configs.train.optimizer.lr},

            {'params': [p for n, p in self.head_b.named_parameters()], 'weight_decay': 0.01, 'lr': self.configs.train.optimizer.lr},
            
            {'params': [p for n, p in self.head_c.named_parameters()], 'weight_decay': 0.01, 'lr': self.configs.train.optimizer.lr},
        ] 
        
        return optimizer_parameters
