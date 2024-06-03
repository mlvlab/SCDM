import math
import os
import torch
import argparse

from imaginaire.config import Config
from imaginaire.evaluation.common2 import compute_all_metrics_data
from imaginaire.utils.dataset import _get_val_dataset_object, _get_data_loader

def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--dataset', required=True,
                        help='dataset name')
    parser.add_argument('--batch_size', '-bs', default=40, type=int)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    fake_b = f'/home/user/imaginaire/configs/celeba_miou/{args.dataset}/fake_b.yaml'
    fake_b_cfg = Config(fake_b)
    bs = args.batch_size

    fake_b_dataset = _get_val_dataset_object(fake_b_cfg)
    data_loader_b = _get_data_loader(fake_b_cfg, fake_b_dataset, bs)
    
    all_metrics = compute_all_metrics_data(data_loader_b=data_loader_b, metrics=['seg_mIOU'], dataset_name='celebamask_hq')
    print(args.dataset)
    print(all_metrics)
    print('---------------')
    
if __name__ == '__main__':
    main()