"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import torch
import argparse
import torch.nn as nn
from torchvision import models
from default_dataset import get_eval_loader


def normalize(x, eps=1e-10):
    return x * torch.rsqrt(torch.sum(x**2, dim=1, keepdim=True) + eps)


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = models.alexnet(pretrained=True).features
        self.channels = []
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                self.channels.append(layer.out_channels)

    def forward(self, x):
        fmaps = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                fmaps.append(x)
        return fmaps


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False))

    def forward(self, x):
        return self.main(x)


class LPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.alexnet = AlexNet()
        self.lpips_weights = nn.ModuleList()
        for channels in self.alexnet.channels:
            self.lpips_weights.append(Conv1x1(channels, 1))
        self._load_lpips_weights()
        # imagenet normalization for range [-1, 1]
        self.mu = torch.tensor([-0.03, -0.088, -0.188]).view(1, 3, 1, 1).cuda()
        self.sigma = torch.tensor([0.458, 0.448, 0.450]).view(1, 3, 1, 1).cuda()

    def _load_lpips_weights(self):
        own_state_dict = self.state_dict()
        if torch.cuda.is_available():
            # state_dict = torch.load('evaluations/alexnet-owt-7be5be79.pth')
            state_dict = torch.load('evaluations/lpips_weights.ckpt')
        else:
            state_dict = torch.load('evaluations/lpips_weights.ckpt',
                                    map_location=torch.device('cpu'))
        for name, param in state_dict.items():
            if name in own_state_dict:
                own_state_dict[name].copy_(param)

    def forward(self, x, y):
        x = (x - self.mu) / self.sigma
        y = (y - self.mu) / self.sigma
        x_fmaps = self.alexnet(x)
        y_fmaps = self.alexnet(y)
        lpips_value = 0
        for x_fmap, y_fmap, conv1x1 in zip(x_fmaps, y_fmaps, self.lpips_weights):
            x_fmap = normalize(x_fmap)
            y_fmap = normalize(y_fmap)
            lpips_value += torch.mean(conv1x1((x_fmap - y_fmap)**2))
        return lpips_value


@torch.no_grad()
def calculate_lpips_given_images(group_of_images):
    # group_of_images = [torch.randn(N, C, H, W) for _ in range(10)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lpips = LPIPS().eval().to(device)
    lpips_values = []
    num_rand_outputs = len(group_of_images)
    # num_rand_outputs = 10

    # calculate the average of pairwise distances among all random outputs
    for i in range(num_rand_outputs-1):
        for j in range(i+1, num_rand_outputs):
            lpips_values.append(lpips(group_of_images[i], group_of_images[j]))
    lpips_value = torch.mean(torch.stack(lpips_values, dim=0))
    return lpips_value.item()


@torch.no_grad()
def calculate_lpips_given_paths(root_path, img_size=256, batch_size=50, test_list=None, seed_list=[]):
    # print('Calculating LPIPS given root path %s' % root_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    paths = [os.path.join(root_path, path) for path in os.listdir(root_path) ] #if path.startswith('seed') and path.split('/')[0][4:] in seed_list]
    loaders = [get_eval_loader(path, img_size, batch_size, shuffle=False, drop_last=False, test_list=test_list) for path in paths]
    loaders_iter = [iter(loader) for loader in loaders]
    num_clips = len(loaders[0])

    lpips_values = []

    for i in range(num_clips):
        group_of_images = [loader.next().cuda() for loader in loaders_iter]
        lpips_values.append(calculate_lpips_given_images(group_of_images))
    lpips_values = torch.tensor(lpips_values)

    lpips_mean = torch.mean(lpips_values)

    return lpips_mean


def make_lpips_list(seed_list, root_path):
    dirr = root_path +'0/samples'
    name = 'lpips_list.txt'
    filelist = os.listdir(dirr)
    filelist.sort()
    with open(name, 'w') as f:
        for seed in seed_list:
            f.writelines(seed + '/samples/' + line + '\n' for line in filelist)


if __name__ == '__main__':
    # python -m metrics.lpips --paths PATH_REAL PATH_FAKE
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default="", help='paths to real and fake images')
    parser.add_argument('--test_list', type=str, default="evaluations/ade20k_list.txt", help='paths to real and fake images')
    parser.add_argument('--img_size', type=int, default=256, help='image resolution')
    parser.add_argument('--batch_size', '-b', type=int, default=50, help='batch size to use')
    parser.add_argument('--seed_list', type=str, default='')
    parser.add_argument('--mini_lpips_size', type=int, default=-1)
    parser.add_argument('--mini_lpips_num', type=int, default=-1)
    args = parser.parse_args()
    if args.mini_lpips_num == -1:
        make_lpips_list(args.seed_list.split(','),args.root_path)
        lpips_value = calculate_lpips_given_paths(args.root_path, args.img_size, args.batch_size, test_list=args.test_list, seed_list = args.seed_list.split(','))
        print('LPIPS: ', lpips_value)
    else:
        import random
        import numpy as np
        results = []
        print_list = []
        for i in range(args.mini_lpips_num):
            print(i)
            seed_list = random.sample(args.seed_list.split(','),args.mini_lpips_size)
            make_lpips_list(seed_list,args.root_path)
            lpips_value = calculate_lpips_given_paths(args.root_path, args.img_size, args.batch_size, test_list=args.test_list, seed_list = seed_list)
            results.append(lpips_value)
            print_list.append('mini batch('+str(args.mini_lpips_size)+') : [' + ','.join(seed_list) + '], LPIPS: '+ str(lpips_value))
        for line in print_list:
            print(line)
        results = np.array(results)
        print('mean : ',results.mean(),', std : ',results.std())
        np.save('mini_lpips_results/mini_lpips_results_'+str(args.mini_lpips_size)+'.npy',results)
