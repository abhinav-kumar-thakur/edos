import os
import json
import torch

class Logger:
    def __init__(self, configs):
        self.configs = configs
        self.dir = os.path.join(configs.logs.dir, configs.title + '-' + configs.task)
        self.state = {
            'kth_fold': None,
            'epoch': None,
            'eval_metrics': []
        }

        if os.path.exists(self.dir):
            self.state = json.load(open(os.path.join(self.dir, self.configs.logs.files.state), 'r'))
        else:
            os.makedirs(self.dir, exist_ok=True)    
            os.makedirs(os.path.join(configs.logs.dir, configs.title + '-' + configs.task, configs.logs.files.models), exist_ok=True)
            json.dump(configs.configs, open(os.path.join(self.dir, 'configs.json'), 'w'))
            json.dump(self.state, open(os.path.join(self.dir, self.configs.logs.files.state), 'w'))

    def log_file(self, log_file, log_dict):
        filepath = os.path.join(self.dir, log_file)
        with open(filepath, 'a+') as f:
            f.write(json.dumps(log_dict))
            f.write('\n')

    def log_text(self, log_file, text):
        filepath = os.path.join(self.dir, log_file)
        with open(filepath, 'a+') as f:
            f.write(text)
            f.write('\n')

    def update_state(self, kth_fold, epoch, model, eval_metrics):
        self.state['kth_fold'] = kth_fold
        self.state['epoch'] = epoch
        json.dump(self.state, open(os.path.join(self.dir, self.configs.logs.files.state), 'w'))
        torch.save(model.state_dict(), os.path.join(self.dir, self.configs.logs.files.models, f'model_current_state.pt'))

    def update_eval_metrics(self, eval_metrics):
        self.state['eval_metrics'].append(eval_metrics)
        json.dump(self.state, open(os.path.join(self.dir, self.configs.logs.files.state), 'w'))

    def get_eval_metrics(self):
        return self.state['eval_metrics']

    def get_state(self):
        return self.state

    def get_current_state_model(self):
        return torch.load(os.path.join(self.dir, self.configs.logs.files.models, f'model_current_state.pt'))

