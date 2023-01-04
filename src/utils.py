def get_args():
    from argparse import ArgumentParser
    
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--config', default='dev.json', required=True, help='Config file from ./configs')
    arg_parser.add_argument('--device', default='cpu', required=True, help='Supported devices: mps/cpu/cuda')
    return arg_parser.parse_args()


def plot_metrics(log_file, y_metrics, x_metrics, filter):
    import json
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    metrics = {
        'eval': [],
        'train': []
    }

    with open(log_file) as f:
        for line in f:
            d = json.loads(line)
            if 'train' in d:
                metrics['train'].append(d)

            if 'eval' in d:
                metrics['eval'].append(d)

    for y_metric in y_metrics:
        y = []
        x = []
        for d in metrics[y_metric[0]]:
            if all([d[k] == v for k, v in filter.items()]):
                y.append(d[y_metric[0]][y_metric[1]][y_metric[2]][y_metric[3]])
                x.append(d[x_metrics[0][1]])
        plt.plot(x, y, label=" ".join(y_metric))
    plt.legend()
    plt.show()
