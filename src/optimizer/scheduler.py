from transformers import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup


def get_scheduler(optimizer, configs):
    if configs.train.scheduler.name == 'cosine':
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=configs.train.scheduler.num_warmup_steps, num_training_steps=configs.train.epochs)
    elif configs.train.scheduler.name == 'cosine_with_hard_restarts':
        return get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=configs.train.scheduler.num_warmup_steps, num_training_steps=configs.train.epochs, num_cycles=configs.train.scheduler.num_cycles)
    else:
        raise ValueError(f'Scheduler {configs.train.scheduler.name} is not supported')