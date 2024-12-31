import os.path as osp
import time
import random
import logging
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
import mlflow
from mlflow.tracking import MlflowClient

from dataset.dataset import MIMICBaseDataset
from torch.utils.data import DataLoader
from model.model import MODELS
from trainer.optimizer import OPTIMIZERS
from trainer.scheduler import SCHEDULERS
from trainer.criterion import CRITERIONS
from metrics.metrics import METRICS
from utils.misc import init_logger, save_params
from utils.args import get_args
from utils.utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def load_all(processed_data_path: str, dataset_name: str):
    processed_data_path = osp.join(processed_data_path, dataset_name)

    train_dataset = read_pickle_file(processed_data_path, f'{dataset_name}_train_data.pickle')
    val_dataset = read_pickle_file(processed_data_path, f'{dataset_name}_val_data.pickle')
    test_dataset = read_pickle_file(processed_data_path, f'{dataset_name}_test_data.pickle')

    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
    }


def get_out_dim(task: str):
    if task == 'mortality_prediction':
        return 1

    elif task == 'los_prediction':
        return 8


def main(args):
    init_logger()
    logger.info(f'Process begins...')
    logger.info(f'Dataset: {args.dataset}')
    logger.info(f'Task: {args.task}')
    logger.info(f'Model: {args.model["name"]}')

    seed_everything(args.seed)

    file_lib = load_all(args.processed_data_path, args.dataset)
    train_dataset = file_lib['train_dataset']
    val_dataset = file_lib['val_dataset']
    test_dataset = file_lib['test_dataset']
    logger.info('Completed file loading')

    train_dataset = MIMICBaseDataset(data=train_dataset,
                                     task=args.task)
    val_dataset = MIMICBaseDataset(data=val_dataset,
                                   task=args.task)
    logger.info('Dataset ready')

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.train_batch_size,
                              shuffle=True)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=args.val_batch_size,
                            shuffle=False)
    
    all_test_loader = []
    test_keys = np.array(list(test_dataset.keys()))
    for idx in range(args.bootstrap_num):
        random.seed(idx)
        np.random.seed(idx)
        sample_indices = np.random.choice(len(test_dataset), len(test_dataset), replace=True)
        sample_keys = test_keys[sample_indices]

        sub_test_dataset = {sample_pos: test_dataset[key] for sample_pos, key in enumerate(sample_keys)}
        sub_test_dataset = MIMICBaseDataset(data=sub_test_dataset,
                                   task=args.task)

        sub_test_loader = DataLoader(dataset=sub_test_dataset,
                                batch_size=args.test_batch_size,
                                shuffle=False)
        
        all_test_loader.append(sub_test_loader)
    logger.info('Dataloader ready')

    out_dim = get_out_dim(args.task)

    global_iter_idx = [0]
    start_epoch = 0

    model_configs = args.model['args'] | {
        'device': device,
        'static_feats_num': args.static_feats_num,
        'intervention_feats_num': args.intervention_feats_num,
        'vital_feats_num': args.vital_feats_num,
        'window_size': args.window_size,
        'out_dim': out_dim,
    }
    model = MODELS[args.model['name']](model_configs)
    model.to(device)
    logger.info('Model ready')

    optimizer = OPTIMIZERS[args.optimizer['name']](model.parameters(), **args.optimizer['args'])
    scheduler = None
    if args.scheduler is not None:
        scheduler = SCHEDULERS[args.scheduler['name']](optimizer, **args.scheduler['args'])
    logger.info('Optimizer and Scheduler ready')

    experiment_name = f'Experiments_{args.model["name"]}'
    mlflow.set_tracking_uri(osp.join(args.log_path, 'mlflow'))
    client = MlflowClient()
    try:
        EXP_ID = client.create_experiment(experiment_name)
    except:
        experiments = client.get_experiment_by_name(experiment_name)
        EXP_ID = experiments.experiment_id

    with mlflow.start_run(experiment_id=EXP_ID,
                          run_name=f'{args.dataset}_{args.task}_{args.model["name"]}_{args.model["mode"]}_{args.start_time}'):
        mlflow.log_params(vars(args))
        model_param_num = sum(p.numel() for p in model.parameters())
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
        mlflow.log_metric('model_param_num', model_param_num)
        mlflow.log_metric('model_size_mb', model_size)

        if args.model['mode'] == 'train':
            early_stopping_counter = 0
            best_score = 0.

            bootstrap_test_results = {}
            bootstrap_test_result_stats = {}
            for metric_name in args.test_metrics:
                if '_K' in metric_name:
                    for k in args.test_metrics[metric_name]['K']:
                        bootstrap_test_results[f'{metric_name}_{k}'] = []
                        bootstrap_test_result_stats['bootstrap_test_stats_' + f'{metric_name}_{k}'] = {'mean': None, 'std': None, 'num': None}
                else:
                    bootstrap_test_results[metric_name] = []
                    bootstrap_test_result_stats['bootstrap_test_stats_' + metric_name] = {'mean': None, 'std': None, 'num': None}

            for epoch_idx in range(start_epoch, args.max_epoch):
                single_train(
                    model,
                    args.task,
                    train_loader,
                    epoch_idx,
                    global_iter_idx,
                    optimizer,
                    criterions=[CRITERIONS[criterion](**args.criterion[criterion]) for criterion in args.criterion],
                    metrics=[METRICS[metric](args.task, **args.val_metrics[metric]) for metric in args.val_metrics],
                    scheduler=scheduler,
                    logging_freq=args.logging_freq)

                if epoch_idx % args.val_freq == 0 or epoch_idx == args.max_epoch - 1:
                    val_results = single_validate(
                        model,
                        args.task,
                        val_loader,
                        epoch_idx,
                        global_iter_idx,
                        criterions=[CRITERIONS[criterion](**args.criterion[criterion]) for criterion in args.criterion],
                        metrics=[METRICS[metric](args.task, **args.val_metrics[metric]) for metric in args.val_metrics])

                    # Early Stopping
                    score = val_results[args.early_stopping_indicator]

                    if score >= best_score:
                        best_model = deepcopy(model)
                        best_optimizer = deepcopy(optimizer)
                        best_scheduler = deepcopy(scheduler)
                        best_score = score
                        best_val_results = val_results
                        best_epoch = epoch_idx
                        best_iter = global_iter_idx[0]
                        early_stopping_counter = 0

                    else:
                        early_stopping_counter += 1

                    if early_stopping_counter >= args.early_stopping_threshold:
                        break
                    elif epoch_idx == args.max_epoch - 1:
                        logger.info(f'Max epoch reached, best epoch is last epoch: {epoch_idx}')
                        break

            for bootstrap_idx, test_loader in enumerate(all_test_loader):
                test_results = single_test(
                    model,
                    args.task,
                    test_loader,
                    bootstrap_idx,
                    bootstrap_idx,
                    criterions=[CRITERIONS[criterion](**args.criterion[criterion]) for criterion in args.criterion],
                    metrics=[
                        METRICS[metric](args.task, **args.test_metrics[metric]) for metric in args.test_metrics
                    ])
                
                for name, result in test_results.items():
                    bootstrap_test_results[name].append(float(result))

            if early_stopping_counter >= args.early_stopping_threshold:
                logger.info(f'Early stopping triggered, best epoch: {best_epoch}')
            elif epoch_idx == args.max_epoch - 1 and early_stopping_counter < args.early_stopping_threshold:
                logger.info(f'Max epoch reached, best epoch: {best_epoch}')

            for metric_name, bootstrap_vals in bootstrap_test_results.items():
                bootstrap_vals = 100*np.array(bootstrap_vals)
                mean = np.round(np.mean(bootstrap_vals), decimals=1)
                std = np.round(np.std(bootstrap_vals), decimals=1)
                num = len(bootstrap_vals)
                bootstrap_test_result_stats['bootstrap_test_stats_' + metric_name]['mean'] = mean.item()
                bootstrap_test_result_stats['bootstrap_test_stats_' + metric_name]['std'] = std.item()
                bootstrap_test_result_stats['bootstrap_test_stats_' + metric_name]['num'] = num
                logger.info(f'Bootstrap {metric_name} mean: {mean} std: {std} num: {num}')

            mlflow.log_params(bootstrap_test_result_stats)

            if args.save_test:
                bootstrap_results_path = osp.join(args.log_path, 'bootstrap_results', args.dataset, args.task,
                                            args.model['name'])
                save_with_pickle(bootstrap_test_results, bootstrap_results_path, f'{args.dataset}_{args.task}_{args.model["name"]}_bootstrap_results_{args.start_time}.pickle')
                save_with_pickle(bootstrap_test_result_stats, bootstrap_results_path, f'{args.dataset}_{args.task}_{args.model["name"]}_bootstrap_result_stats_{args.start_time}.pickle')
                logger.info('Bootstrap results saved')

            if args.save_params:
                save_params(model=best_model,
                            args=args,
                            epoch_idx=best_epoch,
                            iter_idx=best_iter,
                            optimizer=best_optimizer,
                            scheduler=best_scheduler)
                logger.info('Model saved')

            logger.info('Process completed')

        elif args.model['mode'] == 'inference':
            test_results = single_test(
                    model,
                    args.task,
                    test_loader,
                    epoch_idx,
                    global_iter_idx,
                    criterions=[CRITERIONS[criterion](**args.criterion[criterion]) for criterion in args.criterion],
                    metrics=[
                        METRICS[metric](args.task, **args.test_metrics[metric]) for metric in args.test_metrics
                    ])


def single_train(model,
                 task,
                 dataloader,
                 epoch_idx,
                 global_iter_idx,
                 optimizer,
                 criterions=[],
                 metrics=[],
                 scheduler=None,
                 logging_freq=10):
    train_start_time = time.time()
    model.train()
    epoch_loss = []
    prob_all = []
    target_all = []

    for idx, data in enumerate(dataloader):
        x_static = data['static']
        x_dynamic = data['dynamic']
        fourier_coeffs = data['fourier_coeffs']
        labels = data['label']
        x_static = x_static.to(device)
        x_dynamic = x_dynamic.to(device)
        fourier_coeffs = fourier_coeffs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        output = model(x_static=x_static,
                       x_dynamic=x_dynamic,
                       fourier_coeffs=fourier_coeffs)

        out = output['logits']
        reconstructed_vitals = output['reconstructed_vitals']

        loss = 0.
        for criterion in criterions:
            loss += criterion(out, labels)

        epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        if task == 'los_prediction':
            probability = F.softmax(out, dim=-1)
        else:
            probability = torch.sigmoid(out)

        prob_all.append(probability.cpu().detach())
        target_all.append(labels.cpu().detach())

        if idx % logging_freq == 0:
            logger.info(
                f"Epoch: {epoch_idx:4d}, Iteration: {idx:4d} / {len(dataloader):4d} [{global_iter_idx[0]:5d}], Loss: {loss.item()}"
            )
        mlflow.log_metric(key='train_batch_loss', value=loss.item(), step=global_iter_idx[0])

        global_iter_idx[0] += 1

    if scheduler is not None:
        scheduler.step()

    epoch_loss_avg = np.mean(epoch_loss)
    logger.info(f"Epoch: {epoch_idx:4d},  [{global_iter_idx[0]:5d}], Epoch Loss: {epoch_loss_avg}")
    mlflow.log_metrics({
        'train_epoch_time_seconds': time.time() - train_start_time,
        'train_epoch_loss': epoch_loss_avg
    },
                       step=epoch_idx)

    prob_all = np.concatenate(prob_all, axis=0)
    target_all = np.concatenate(target_all, axis=0)

    for metric in metrics:
        score = metric.calculate(prob_all, target_all)
        if isinstance(score, list):
            for idx, s in enumerate(score):
                mlflow.log_metric(key=f'train_{metric.NAME}_{metric.K[idx]}', value=s, step=epoch_idx)
        else:
            mlflow.log_metric(key=f'train_{metric.NAME}', value=score, step=epoch_idx)


def single_validate(model, task, dataloader, epoch_idx, global_iter_idx, criterions=[], metrics=[]):
    model.eval()
    epoch_loss = []
    prob_all = []
    target_all = []

    for _, data in enumerate(tqdm(dataloader)):
        x_static = data['static']
        x_dynamic = data['dynamic']
        fourier_coeffs = data['fourier_coeffs']
        labels = data['label']
        x_static = x_static.to(device)
        x_dynamic = x_dynamic.to(device)
        fourier_coeffs = fourier_coeffs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            output = model(x_static=x_static,
                       x_dynamic=x_dynamic,
                       fourier_coeffs=fourier_coeffs)

            out = output['logits']
            reconstructed_vitals = output['reconstructed_vitals']

            loss = 0.
            for criterion in criterions:
                loss += criterion(out, labels)

            epoch_loss.append(loss.item())

            if task == 'los_prediction':
                probability = F.softmax(out, dim=-1)
            else:
                probability = torch.sigmoid(out)

            prob_all.append(probability.cpu())
            target_all.append(labels.cpu())

    results = {}

    epoch_loss_avg = np.mean(epoch_loss)
    logger.info(f"Epoch: {epoch_idx:4d},  [{global_iter_idx[0]:5d}], Epoch Loss: {epoch_loss_avg}")
    mlflow.log_metric(key='val_epoch_loss', value=epoch_loss_avg, step=epoch_idx)

    prob_all = np.concatenate(prob_all, axis=0)
    target_all = np.concatenate(target_all, axis=0)

    for metric in metrics:
        score = metric.calculate(prob_all, target_all)
        if isinstance(score, list):
            for idx, s in enumerate(score):
                mlflow.log_metric(key=f'val_{metric.NAME}_{metric.K[idx]}', value=s, step=epoch_idx)
                results[f'{metric.NAME}_{metric.K[idx]}'] = s
        else:
            mlflow.log_metric(key=f'val_{metric.NAME}', value=score, step=epoch_idx)
            results[metric.NAME] = score

    return results


def single_test(model, task, dataloader, epoch_idx, global_iter_idx, criterions=[], metrics=[]):
    model.eval()
    epoch_loss = []
    prob_all = []
    target_all = []

    for _, data in enumerate(tqdm(dataloader)):
        x_static = data['static']
        x_dynamic = data['dynamic']
        fourier_coeffs = data['fourier_coeffs']
        labels = data['label']
        x_static = x_static.to(device)
        x_dynamic = x_dynamic.to(device)
        fourier_coeffs = fourier_coeffs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            output = model(x_static=x_static,
                       x_dynamic=x_dynamic,
                       fourier_coeffs=fourier_coeffs)

            out = output['logits']
            reconstructed_vitals = output['reconstructed_vitals']

            loss = 0.
            for criterion in criterions:
                loss += criterion(out, labels)

            epoch_loss.append(loss.item())

            if task == 'los_prediction':
                probability = F.softmax(out, dim=-1)
            else:
                probability = torch.sigmoid(out)

            prob_all.append(probability.cpu())
            target_all.append(labels.cpu())

    results = {}

    epoch_loss_avg = np.mean(epoch_loss)
    logger.info(f"Epoch: {epoch_idx:4d},  [{global_iter_idx:5d}], Epoch Loss: {epoch_loss_avg}")
    mlflow.log_metric(key='test_epoch_loss', value=epoch_loss_avg, step=epoch_idx)

    prob_all = np.concatenate(prob_all, axis=0)
    target_all = np.concatenate(target_all, axis=0)

    for metric in metrics:
        score = metric.calculate(prob_all, target_all)
        if isinstance(score, list):
            for idx, s in enumerate(score):
                mlflow.log_metric(key=f'test_{metric.NAME}_{metric.K[idx]}', value=s, step=epoch_idx)
                results[f'{metric.NAME}_{metric.K[idx]}'] = s
        else:
            mlflow.log_metric(key=f'test_{metric.NAME}', value=score, step=epoch_idx)
            results[metric.NAME] = score

    return results


if __name__ == '__main__':
    args = get_args()
    main(args=args)
