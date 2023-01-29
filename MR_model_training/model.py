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

class MLP(nn.Module):
    def __init__(self, nInput, nOutput, nLayer, nHidden, act_fn):
        super(MLP, self).__init__()
        layers = []

        ##### implement this part #####
        for i in range(nLayer-1):
            if i == 0:
                layer = nn.Linear(nInput, nHidden)
            else:
                layer = nn.Linear(nHidden, nHidden)
            layers.append(layer)
            layers.append(act_fn)
        layer = nn.Linear(nHidden, nOutput)
        layers.append(layer)
        ###############################

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.model(x)