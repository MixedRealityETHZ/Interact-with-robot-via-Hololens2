import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms

import pickle
from tqdm import tqdm
import os
from torch.utils import data
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import glob
import numpy as np

action_dict={
    'TurnRight':0,
    'TurnLeft':1,
    'WalkForward':2,
    'Stop':3
}
class HandGestureDataset(data.Dataset):
    def __init__(self, datadir='data', skipfirstframes=30, recognition_frames=10, mode=''):
        self.mode=mode

        self.skipfirstframes = skipfirstframes
        self.datadir = datadir
        files = sorted(glob.glob('data/*.txt'))
        all_hand_joints=[]
        all_actions=[]
        for file in files:
            with open(file, "r") as f:
                lines=f.readlines()
            lines_np=[]
            for line in lines[skipfirstframes:100]:
                line=line.replace(')(', ',')
                line=line.replace('(', '')
                line=line.replace(')', '')
                line_np=np.array(list(map(float, line.split(','))))
                lines_np.append(line_np)
            hand_joints=np.stack(lines_np)
            hand_joints=hand_joints.reshape(-1, 78)
            all_hand_joints.append(hand_joints)
            
            action_string=file.split('_')[-1].split('.')[0]
            action = action_dict[action_string]
            all_actions.append(np.array([action]))
            # print(action)
            # print(all_actions)
            # exit(0)
            # print(hand_joints.shape)
            # for frameid, perframe_pts in enumerate(hand_joints):
            #     perframe_pts_tri = trimesh.PointCloud(vertices=perframe_pts)
            #     perframe_pts_tri.export(f'tmp/{frameid:04d}.ply')
        all_actions=torch.from_numpy(np.concatenate(all_actions)).float()
        all_hand_joints=np.stack(all_hand_joints)
        # print(all_actions.shape)
        # print(all_hand_joints.shape)
        self.recognition_frames=recognition_frames
        self.all_hand_joints=all_hand_joints.reshape(-1, self.recognition_frames*78)
        # print(self.all_hand_joints.shape)
        all_actions=all_actions.unsqueeze(-1).repeat(1, (100-skipfirstframes)//recognition_frames)
        # print(all_actions)
        # print(all_actions.shape)
        all_actions=all_actions.reshape(-1)
        
        # exit(0)
        
        # all_actions=F.one_hot(all_actions.long(), 4)
        self.all_actions=all_actions.long()
        
        
    def __len__(self):
        if self.mode=='train':
            return int(self.all_hand_joints.shape[0]*0.75)
        elif self.mode=='test':
            return int(self.all_hand_joints.shape[0]*0.25)
        else:
            return self.all_hand_joints.shape[0]

    def __getitem__(self, index):
        if self.mode=='test':
            # print('tot length', self.all_hand_joints.shape[0])
            index+=int(self.all_hand_joints.shape[0]*0.75)
            # print(index)


        input=torch.from_numpy(self.all_hand_joints[index]).float()
        label=self.all_actions[index]
        # print(input.shape)
        # print(label.shape)
        return input, label











