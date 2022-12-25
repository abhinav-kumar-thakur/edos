from src.optimizer.adafactor import AdaFactorOptimizer
from src.optimizer.adam import AdamOptimizer


def get_optimizer(model, configs):
    optimizer_name = configs.train.optimizer.name
    if optimizer_name == 'Adam':
        return AdamOptimizer(model, 0.00001)
    elif optimizer_name == 'Adafactor':
        return AdaFactorOptimizer(model, None)
    else:
        raise ValueError(f'Optimizer {optimizer_name} is not supported')
