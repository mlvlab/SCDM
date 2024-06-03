import torch as th
from guided_diffusion.image_datasets import load_data
from guided_diffusion import dist_util, logger
import torchvision as tv
import os 
import cv2
import random

def get_edges(t):
    edge = th.ByteTensor(t.size()).zero_()
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float()

dist_util.setup_dist()

dataset_mode = 'ade20k'
data_dir = 'data/ADE20K/ADEChallengeData2016'
new_dir = 'data/ADE20K_noisy/ADE20K_edge'
batch_size = 50
image_size = 256
class_cond = True

data = load_data(
        dataset_mode=dataset_mode,
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=class_cond,
        deterministic=True,
        random_crop=False,
        random_flip=False,
        is_train=False
    )


for i , (batch, cond) in enumerate(data):
    if 'instance' in cond:
        inst_map = cond['instance']
        instance_edge_map = get_edges(inst_map)
        instance_edge_map = instance_edge_map.permute(0,2,3,1)
        map_shape = instance_edge_map.shape
        new_edge_map = th.ByteTensor(inst_map.size()).zero_().float().permute(0,2,3,1)

        for b in range(map_shape[0]):
            for x in range(map_shape[1]):
                for y in range(map_shape[2]):
                    if instance_edge_map[b][x][y] == 1:
                        rad = 2
                        for dx in range(-rad,rad+1):
                            for dy in range(-(rad-abs(dx)),(rad-abs(dx))+1):
                                if x+dx >= 0 and x+dx <map_shape[1]:
                                    if y+dy >= 0 and y+dy <map_shape[2]:
                                        new_edge_map[b][x+dx][y+dy] = 1

        instance_edge_map = new_edge_map

        instance_edge_map = instance_edge_map.permute(0,3,1,2)
        edge_unlabeled = instance_edge_map * (-1) + 1
        newlabel = cond['label_ori']
        newlabel = newlabel * edge_unlabeled.squeeze(1)
        for j in range(inst_map.shape[0]):
            save_dir = os.path.join(new_dir, cond['path'][j].split('/')[-1].split('.')[0] + '.png')
            cv2.imwrite(save_dir, newlabel[j].cpu().numpy())
    if i >= (2000/batch_size):
        break