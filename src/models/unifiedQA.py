import torch as t
from transformers import T5Tokenizer, T5ForConditionalGeneration


class UnifiedQAClassifier(t.nn.Module):
    def __init__(self, configs, device='cpu') -> None:
        super().__init__()

        self.device = device
        self.configs = configs
        self.model_name = f'allenai/unifiedqa-v2-t5-{configs.model.unifiedQA.model_size}-1251000'
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(device)

    def forward(self, input, train=True):
        encoding = self.tokenizer(input['question'], padding="longest", max_length=512, truncation=True, return_tensors="pt")
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

        loss = None
        predictions = None
        if train:
            target_encoding = self.tokenizer(input['label_sexist'], padding="longest", max_length=10, truncation=True)
            labels = target_encoding.input_ids

            labels = t.tensor(labels)
            labels[labels == self.tokenizer.pad_token_id] = -100

            loss = self.model(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device), labels=labels.to(self.device)).loss
        else:
            res = self.model.generate(input_ids.to(self.device))
            predictions = self.tokenizer.batch_decode(res, skip_special_tokens=True)

        return predictions, loss

    def predict(self, input):
        encoding = self.tokenizer(input['question'], padding="longest", max_length=512, truncation=True, return_tensors="pt")
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

        output = self.model.generate(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device), max_length=4)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
