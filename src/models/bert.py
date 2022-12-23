import torch as t
from torch.optim import Adam
from transformers import AutoModel


class BertClassifier(t.nn.Module):
    def __init__(self, configs, device='cpu') -> None:
        super().__init__()

        self.device = device
        self.configs = configs
        self.bert = AutoModel.from_pretrained(configs.model.bert.name).to(device)
        self.linear_one = t.nn.Linear(configs.model.bert.dimentions, 512).to(device)
        self.linear_two = t.nn.Linear(512, 256).to(device)
        self.head_a = t.nn.Linear(256, len(self.configs.datasets.label_sexist.configs)).to(device)

        class_weights = t.FloatTensor([1, 3]).to(device)
        self.loss = t.nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = Adam(self.parameters(), lr=0.0001)
        self.label2idx = self.configs.datasets.label_sexist_ids.configs
        self.idx2label = {v: k for k, v in self.label2idx.items()}

    def forward(self, input, train=True):
        input_ids = input['input_ids'].to(self.device)
        attention_mask = input['attention_mask'].to(self.device)

        with t.no_grad():
            _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)

        x = t.relu(self.linear_one(pooled_output))
        x = t.relu(self.linear_two(x))
        pred = self.head_a(x)
        
        loss = None
        if train:
            actaul = [self.label2idx[l] for l in  input['label_sexist']]
            actaul = t.tensor(actaul).to(self.device)
            loss = self.loss(pred, actaul)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        pred_ids = t.argmax(pred, dim=1)
        labels = [self.idx2label[p.item()] for p in pred_ids]
        return labels, loss.item() if loss else None
