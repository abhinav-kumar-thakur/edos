from argparse import ArgumentParser


def get_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--config', default='dev.json', required=True, help='Config file from ./configs')
    arg_parser.add_argument('--device', default='cpu', required=True, help='Supported devices: mps/cpu/cuda')
    return arg_parser.parse_args()
