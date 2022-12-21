import os
import json

class Logger:
    def __init__(self, configs):
        self.configs = configs
        self.dir = os.path.join(configs.logs.dir, configs.title + '-' + configs.task)

        if not os.path.exists(self.dir):
            os.makedirs(self.dir, exist_ok=True)
            os.makedirs(os.path.join(configs.logs.dir, configs.title + '-' + configs.task, configs.logs.files.models), exist_ok=True)
            json.dump(configs.configs, open(os.path.join(self.dir, 'configs.json'), 'w'))
        

    def log_file(self, log_file, log_dict):
        filepath = os.path.join(self.dir, log_file)
        with open(filepath, 'a+') as f:
            f.write(json.dumps(log_dict))
            f.write('\n')

    def log_csv(self, log_file, data):
        filepath = os.path.join(self.dir, log_file)
        with open(filepath, 'w') as f:
            for line in data:
                row = [str(x).replace('"',"'") for x in line]
                f.write(','.join([f'"{x}"' for x in row]))
                f.write('\n')

    def log_text(self, log_file, text):
        filepath = os.path.join(self.dir, log_file)
        with open(filepath, 'a+') as f:
            f.write(text)
            f.write('\n')

    def log(self, text):
        print('----------------------------------------')
        print(text)
        print('----------------------------------------')
