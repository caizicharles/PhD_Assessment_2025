import argparse
import yaml
from .misc import get_time_str


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='mimiciii')
    parser.add_argument('--task', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_test', type=bool, default=False)
    parser.add_argument('--save_params', type=bool, default=False)

    parser.add_argument('--processed_data_path', type=str, default='')
    parser.add_argument('--log_path', type=str, default='')

    optimizer_hyp_search_args = {
        'lr': 0.0001,
        'weight_decay': 0.,
    }

    scheduler_hyp_search_args = {'gamma': 0.995}

    model_hyp_search_args = {
        'scoring_hidden_dim': 64,
        'k_coeffs': 4,
        'static_hidden_sizes': [32, 64],
        'fuse_dim': 64,
        'dynamic_hidden_dim': 64,
        'dynamic_layers': 3,
        'predictor_hidden_sizes': [32, 16],
        'activation': 'relu',
        'dropout': 0.1,
        'softmax_temp': 1.
    }

    for hyp, default_val in optimizer_hyp_search_args.items():
        parser.add_argument(f'--{hyp}', type=type(default_val), default=default_val)
    for hyp, default_val in scheduler_hyp_search_args.items():
        parser.add_argument(f'--{hyp}', type=type(default_val), default=default_val)
    for hyp, default_val in model_hyp_search_args.items():
        parser.add_argument(f'--{hyp}', type=type(default_val), default=default_val)

    # Config File
    config_parser = argparse.ArgumentParser(description='Algorithm Config', add_help=False)
    config_parser.add_argument('-c', '--config', default=None, type=str, help='YAML config file')

    args_config, remaining = config_parser.parse_known_args()
    assert args_config.config is not None, 'Config file must be specified'

    with open(args_config.config, 'r') as f:
        cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)

    if args.task != 'data_preparation':
        for hyp in optimizer_hyp_search_args.keys():
            value = getattr(args, hyp, None)
            if value is not None:
                optimizer_hyp_search_args[hyp] = value

        for hyp in scheduler_hyp_search_args.keys():
            value = getattr(args, hyp, None)
            if value is not None:
                scheduler_hyp_search_args[hyp] = value

        for hyp in model_hyp_search_args.keys():
            value = getattr(args, hyp, None)
            if value is not None:
                model_hyp_search_args[hyp] = value

        args.optimizer['args'] = optimizer_hyp_search_args
        args.scheduler['args'] = scheduler_hyp_search_args
        args.model['args'] = model_hyp_search_args

    args.start_time = get_time_str()

    return args


if __name__ == '__main__':
    args = get_args()
    print(args)
