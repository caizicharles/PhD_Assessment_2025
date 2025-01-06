import sys
import os.path as osp
import yaml
import random
import logging
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dataset.dataset import MIMICBaseDataset
from torch.utils.data import DataLoader
from model.model import MODELS
from utils.misc import init_logger
from utils.args import get_args
from utils.utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger()


class Args:

    def __init__(self, **entries):
        self.__dict__.update(entries)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)


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
        return 6


def run_visualization(args):

    init_logger()

    ckpt = torch.load(args.checkpoint, map_location=device)
    args = ckpt['args']
    args = yaml.safe_load(args)
    args = Args(**args)

    seed_everything(args.seed)

    file_lib = load_all(args.processed_data_path, args.dataset)
    original_test_dataset = file_lib['test_dataset']
    logger.info('Completed file loading')

    test_dataset = MIMICBaseDataset(data=original_test_dataset, task=args.task)
    logger.info('Dataset ready')

    test_loader = DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, shuffle=False)
    logger.info('Dataloader ready')

    out_dim = get_out_dim(args.task)

    model_configs = args.model['args'] | {
        'static_feats_num': args.static_feats_num,
        'intervention_feats_num': args.intervention_feats_num,
        'vital_feats_num': args.vital_feats_num,
        'window_size': args.window_size,
        'out_dim': out_dim,
    }
    model = MODELS[args.model['name']](model_configs)
    model.to(device)

    logger.info(f"Load pretrained model from {args.checkpoint}")
    model.load_state_dict(ckpt['model'])
    logger.info('Model ready')

    visualization_data = single_test(
                model,
                args.task,
                test_loader)
    
    subject_id = visualization_data['subject_id'].squeeze()
    order_id = visualization_data['order_id'].squeeze()
    prob = visualization_data['probabilities'].squeeze()
    labels = visualization_data['labels'].squeeze()
    reconsturcted_vitals = visualization_data['reconsturcted_vitals']

    pred = prob >= 0.5
    pred = pred.astype(int)
    
    mask = labels == 1
    pos_subjects = np.unique(subject_id[mask])
    masked_pred = pred[mask]

    flag = 0
    for subject in pos_subjects:
        subject_mask = np.where(subject_id == subject)[0]
        masked_pred = pred[subject_mask]
        masked_prob = prob[subject_mask]

        if len(set(masked_pred)) == 2:
            patient_data = original_test_dataset[subject]
            
            print(subject, masked_pred)
            print('-'*50)
            
            for idx, mpred in enumerate(masked_pred):
                if mpred == 1:
                    t = args.window_increment*idx + np.arange(args.window_size)
                    hr = patient_data['vitals_labs_mean']['hr']['windowed_data'][idx]
                    dbp = patient_data['vitals_labs_mean']['diastolic_bp']['windowed_data'][idx]
                    mbp = patient_data['vitals_labs_mean']['mean_bp']['windowed_data'][idx]

                    plt.plot(t, hr, label='heart rate', color='cornflowerblue')
                    plt.axvspan(t.max() - args.window_increment, t.max(), color='pink', alpha=0.3, label='new input region')
                    plt.xlabel('time', fontsize=14)
                    plt.xticks(fontsize=12)
                    plt.yticks(fontsize=12)
                    plt.legend(fontsize=12)
                    # plt.show()
                    plt.savefig(osp.join(args.log_data_path, 'visualization_figures/windowed_hr.png'))
                    plt.clf()

                    plt.plot(t, dbp, label='diastolic blood pressure', color='orange')
                    plt.plot(t, mbp, label='mean blood pressure', color='red')
                    plt.axvspan(t.max() - args.window_increment, t.max(), color='pink', alpha=0.3, label='new input region')
                    plt.xlabel('time', fontsize=14)
                    plt.xticks(fontsize=12)
                    plt.yticks(fontsize=12)
                    plt.legend(fontsize=12)
                    # plt.show()
                    plt.savefig(osp.join(args.log_data_path, 'visualization_figures/windowed_bp.png'))
                    plt.clf()

                    flag = 1
                    break
            if flag == 1:
                plt.plot(masked_prob, label='probability', color='cornflowerblue')
                plt.axhspan(0, 0.5, color='limegreen', alpha=0.3)
                plt.axhspan(0.5, 1, color='pink', alpha=0.3)
                plt.xlabel('window index', fontsize=14)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.legend(fontsize=12)
                # plt.show()
                plt.savefig(osp.join(args.log_data_path, 'visualization_figures/probability.png'))

                original_hr = patient_data['vitals_labs_mean']['hr']['original_data']
                original_dbp = patient_data['vitals_labs_mean']['diastolic_bp']['original_data']
                original_mbp = patient_data['vitals_labs_mean']['mean_bp']['original_data']

                subject_reconstructed_vitals = reconsturcted_vitals[subject_mask]
                windowed_reconstructed_dbp = subject_reconstructed_vitals[:, 0]
                windowed_reconstructed_mbp = subject_reconstructed_vitals[:, 1]
                windowed_reconstructed_hr = subject_reconstructed_vitals[:, 2]

                reconstructed_dbp = np.concatenate((windowed_reconstructed_dbp[:, :12].flatten(), windowed_reconstructed_dbp[-1, 12:]))
                plt.plot(original_dbp, label='original diastolic blood pressure')
                plt.plot(reconstructed_dbp, label='reconstructed diastolic blood pressure')

                for i, status in enumerate(masked_pred):
                    start_idx = i * args.window_increment
                    end_idx = start_idx + args.window_size

                    if i != 0:
                        start_idx = end_idx - args.window_increment

                    if end_idx > len(original_dbp):
                        end_idx = len(original_dbp)
                    
                    if status == 1:
                        plt.axvspan(start_idx, end_idx, color='pink', alpha=0.3)
                    else:
                        plt.axvspan(start_idx, end_idx, color='limegreen', alpha=0.3)
                
                plt.xlabel('time', fontsize=14)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.legend(fontsize=12)
                plt.legend(fontsize=12)
                # plt.show()
                plt.savefig(osp.join(args.log_data_path, 'visualization_figures/reconstructed_dbp.png'))
                plt.clf()

                reconstructed_mbp = np.concatenate((windowed_reconstructed_mbp[:, :12].flatten(), windowed_reconstructed_mbp[-1, 12:]))
                plt.plot(original_mbp, label='original mean blood pressure')
                plt.plot(reconstructed_mbp, label='reconstructed mean blood pressure')
                
                for i, status in enumerate(masked_pred):
                    start_idx = i * args.window_increment
                    end_idx = start_idx + args.window_size

                    if i != 0:
                        start_idx = end_idx - args.window_increment

                    if end_idx > len(original_dbp):
                        end_idx = len(original_dbp)
                    
                    if status == 1:
                        plt.axvspan(start_idx, end_idx, color='pink', alpha=0.3)
                    else:
                        plt.axvspan(start_idx, end_idx, color='limegreen', alpha=0.3)
                
                plt.xlabel('time', fontsize=14)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.legend(fontsize=12)
                plt.legend(fontsize=12)
                # plt.show()
                plt.savefig(osp.join(args.log_data_path, 'visualization_figures/reconstructed_mbp.png'))
                plt.clf()

                reconstructed_hr = np.concatenate((windowed_reconstructed_hr[:, :12].flatten(), windowed_reconstructed_hr[-1, 12:]))
                plt.plot(original_hr, label='original heart rate')
                plt.plot(reconstructed_hr, label='reconstructed heart rate')
                
                for i, status in enumerate(masked_pred):
                    start_idx = i * args.window_increment
                    end_idx = start_idx + args.window_size

                    if i != 0:
                        start_idx = end_idx - args.window_increment

                    if end_idx > len(original_dbp):
                        end_idx = len(original_dbp)
                    
                    if status == 1:
                        plt.axvspan(start_idx, end_idx, color='pink', alpha=0.3)
                    else:
                        plt.axvspan(start_idx, end_idx, color='limegreen', alpha=0.3)
                
                plt.xlabel('time', fontsize=14)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.legend(fontsize=12)
                plt.legend(fontsize=12)
                # plt.show()
                plt.savefig(osp.join(args.log_data_path, 'visualization_figures/reconstructed_hr.png'))
                plt.clf()
                break


def single_test(model, task, dataloader):
    model.eval()
    prob_all = []
    target_all = []
    visualization_data = {
        'subject_id': [],
        'order_id': [],
        'probabilities': None,
        'labels': None,
        'reconsturcted_vitals': []
    }

    for _, data in enumerate(tqdm(dataloader)):
        subject_id = data['subject_id']
        order_id = data['order_id']
        x_static = data['static']
        x_dynamic = data['dynamic']
        fourier_coeffs = data['fourier_coeffs']
        labels = data['label']
        x_static = x_static.to(device)
        x_dynamic = x_dynamic.to(device)
        fourier_coeffs = fourier_coeffs.to(device)
        labels = labels.to(device)

        visualization_data['subject_id'].append(subject_id)
        visualization_data['order_id'].append(order_id)

        with torch.no_grad():
            output = model(x_static=x_static, x_dynamic=x_dynamic, fourier_coeffs=fourier_coeffs)

            out = output['logits']
            reconstructed_vitals = output['reconstructed_vitals']
            visualization_data['reconsturcted_vitals'].append(reconstructed_vitals.cpu())

            if task == 'los_prediction':
                probability = F.softmax(out, dim=-1)
            else:
                probability = torch.sigmoid(out)

            prob_all.append(probability.cpu())
            target_all.append(labels.cpu())

    prob_all = np.concatenate(prob_all, axis=0)
    target_all = np.concatenate(target_all, axis=0)
    visualization_data['probabilities'] = prob_all
    visualization_data['labels'] = target_all
    visualization_data['subject_id'] = torch.cat(visualization_data['subject_id'], dim=0).numpy()
    visualization_data['order_id'] = torch.cat(visualization_data['order_id'], dim=0).numpy()
    visualization_data['reconsturcted_vitals'] = torch.cat(visualization_data['reconsturcted_vitals'], dim=0).numpy()

    return visualization_data


if __name__ == '__main__':
    args = get_args()
    run_visualization(args=args)
