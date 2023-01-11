import torch as t
from transformers import AutoModel

from src.lossFunctions.focal_loss import FocalLoss

# min max normalize tensor
def min_max_normalize(x, dim=0):
    y = (x - x.min(dim=dim, keepdim=True)[0])
    return y / t.sum(y)

class BertClassifier(t.nn.Module):
    def __init__(self, configs, device='cpu') -> None:
        super().__init__()

        self.device = device
        self.configs = configs

        task_a_loss_weights = t.FloatTensor(self.configs.model.bert.heads.a.loss_weights).to(device)
        self.loss_a = FocalLoss(alpha=task_a_loss_weights, gamma=2) if self.configs.train.loss == 'fl'else t.nn.CrossEntropyLoss()
        self.label2idx_a = self.get_label_index_a()
        self.idx2label_a = {v: k for k, v in self.label2idx_a.items()}

        task_b_loss_weights = t.FloatTensor(self.configs.model.bert.heads.b.loss_weights).to(device)
        self.loss_b = FocalLoss(alpha=task_b_loss_weights) if self.configs.train.loss == 'fl'else t.nn.CrossEntropyLoss()
        self.label2idx_b = self.get_label_index_b()
        self.idx2label_b = {v: k for k, v in self.label2idx_b.items()}

        task_c_loss_weights = t.FloatTensor(self.configs.model.bert.heads.c.loss_weights).to(device)
        self.loss_c = FocalLoss(alpha=task_c_loss_weights) if self.configs.train.loss == 'fl'else t.nn.CrossEntropyLoss()
        self.label2idx_c = self.get_label_index_c()
        self.idx2label_c = {v: k for k, v in self.label2idx_c.items()}

        self.bert = AutoModel.from_pretrained(configs.model.bert.name).to(device)
        self.hidden_layer = t.nn.Linear(configs.model.bert.dimensions, 256).to(device)
        self.head_c = t.nn.Linear(256, len(self.label2idx_c)).to(device)
        self.head_b = t.nn.Linear(len(self.label2idx_c), len(self.label2idx_b)).to(device)
        self.head_a = t.nn.Linear(len(self.label2idx_b), len(self.label2idx_a)).to(device)

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
        x = t.relu(self.hidden_layer(x))
        pred_c = self.head_c(x)
        x_c = t.relu(pred_c)
        pred_b = self.head_b(x_c)
        x_b = t.relu(pred_b)
        pred_a = self.head_a(x_b)
        
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

        labels = {}
        pred_a_ids = t.argmax(pred_a, dim=1)
        pred_b_ids = t.argmax(pred_b[:, 1:], dim=1)
        pred_c_ids = t.argmax(pred_c[:, 1:], dim=1)
        for i in range(len(pred_a_ids)):
            sexist_label = self.idx2label_a[pred_a_ids[i].item()]
            labels[batch['rewire_id'][i]] = {
                'sexist': sexist_label if 'a' in self.configs.train.task else None,
                'category': self.idx2label_b[pred_b_ids[i].item() + 1] if 'b' in self.configs.train.task else None,
                'vector': self.idx2label_c[pred_c_ids[i].item() + 1] if 'c' in self.configs.train.task else None,
                'scores': {
                    'sexist': {k: v.item() for k, v in zip(self.idx2label_a.values(), pred_a[i])}  if 'a' in self.configs.train.task else None,
                    'category': {k: v.item() for k, v in zip(self.idx2label_b.values(), pred_b[i])} if 'b' in self.configs.train.task else None,
                    'vector': {k: v.item() for k, v in zip(self.idx2label_c.values(), pred_c[i])} if 'c' in self.configs.train.task else None
                },
                'confidence': {
                    'sexist': t.abs(t.diff(t.topk(min_max_normalize(pred_a[i]), 2)[0])).item() if 'a' in self.configs.train.task else None,
                    'category': t.abs(t.diff(t.topk(min_max_normalize(pred_b[i, 1:]), 2)[0])).item() if 'b' in self.configs.train.task else None,
                    'vector': t.abs(t.diff(t.topk(min_max_normalize(pred_c[i, 1:]), 2)[0])).item() if 'c' in self.configs.train.task else None
                },
                'confidence_s': {
                    'sexist': t.abs(t.diff(t.topk(t.softmax(pred_a[i], dim=0), 2)[0])).item() if 'a' in self.configs.train.task else None,
                    'category': t.abs(t.diff(t.topk(t.softmax(pred_b[i, 1:], dim=0), 2)[0])).item() if 'b' in self.configs.train.task else None,
                    'vector': t.abs(t.diff(t.topk(t.softmax(pred_c[i, 1:], dim=0), 2)[0])).item() if 'c' in self.configs.train.task else None
                },
                'uncertainity': {
                    'sexist': -t.sum(t.softmax(pred_a[i], dim=0) * t.log_softmax(pred_a[i], dim=0)).item() if 'a' in self.configs.train.task else None,
                    'category': -t.sum(t.softmax(pred_b[i, 1:], dim=0) * t.log_softmax(pred_b[i, 1:], dim=0)).item() if 'b' in self.configs.train.task else None,
                    'vector': -t.sum(t.softmax(pred_c[i, 1:], dim=0) * t.log_softmax(pred_c[i, 1:], dim=0)).item() if 'c' in self.configs.train.task else None
                }
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
    
    def get_trainable_parameters_with_LLRD(self, init_lr, llrd_factor):
        """
        * init_lr: max lr, to be used in classifiaction heads and bert pooler layer
        * llrd_decay: Layer-wise Learning Rate Decay factor, (should be smaller than one, e.g. 0.8)
        """
        # Layer-wise Laerning Rate Decay (LLRD)
        optimizer_params = []
        model_params = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        lr = init_lr
        # ===  Classification heads and Bert Pooler layer ======================================================  
        pool_and_clf_layers = ["pooler","head_a","head_b","head_c","hidden_layer"]
        params_0 = [p for n,p in model_params if any(nd in n for nd in pool_and_clf_layers) 
                    and any(nd in n for nd in no_decay)]
        params_1 = [p for n,p in model_params if any(nd in n for nd in pool_and_clf_layers)
                    and not any(nd in n for nd in no_decay)]
        
        head_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}    
        optimizer_params.append(head_params)
            
        head_params = {"params": params_1, "lr": lr, "weight_decay": 0.01}    
        optimizer_params.append(head_params)
                    
        # === 12 Hidden layers of Bert ==========================================================
        
        for layer in range(11,-1,-1):        
            params_0 = [p for n,p in model_params if f"encoder.layer.{layer}." in n 
                        and any(nd in n for nd in no_decay)]
            params_1 = [p for n,p in model_params if f"encoder.layer.{layer}." in n 
                        and not any(nd in n for nd in no_decay)]
            
            layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
            optimizer_params.append(layer_params)   
                                
            layer_params = {"params": params_1, "lr": lr, "weight_decay": 0.01}
            optimizer_params.append(layer_params)       
            
            lr *= llrd_factor     
            
        # === Embeddings layer of Bert ==========================================================
        
        params_0 = [p for n,p in model_params if "embeddings" in n 
                    and any(nd in n for nd in no_decay)]
        params_1 = [p for n,p in model_params if "embeddings" in n
                    and not any(nd in n for nd in no_decay)]
        
        embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0} 
        optimizer_params.append(embed_params)
            
        embed_params = {"params": params_1, "lr": lr, "weight_decay": 0.01} 
        optimizer_params.append(embed_params)  

        return optimizer_params