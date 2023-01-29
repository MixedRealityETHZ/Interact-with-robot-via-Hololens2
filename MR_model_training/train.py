import time
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
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
from torch.utils.data import DataLoader
from dataset import HandGestureDataset
from model import MLP

def test(model, dl_test, device='cuda:0'):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dl_test:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(dl_test.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(dl_test.dataset),
        100. * correct / len(dl_test.dataset)))
    
def train(model, dl_train, optimizer, epoch, log_interval=100, device='cuda:0'):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(dl_train):
        data, target = data.to(device), target.to(device)
        
        # first we need to zero the gradient, otherwise PyTorch would accumulate them
        optimizer.zero_grad()         
        
        ##### implement this part #####
        output = model(data)
        # print(data.shape)
        # print(output.shape)
        # print(target.shape)
        # print(target)
        # print(output)
        # exit(0)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        ###############################

        # stats
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dl_train.dataset),
                100. * batch_idx / len(dl_train), loss.item()))

    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        loss, correct, len(dl_train.dataset),
        100. * correct / len(dl_train.dataset)))

recognition_frames=5

dataset=HandGestureDataset(recognition_frames=recognition_frames)
train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.7), len(dataset)-int(len(dataset)*0.7)])

# train_set=HandGestureDataset(recognition_frames=recognition_frames, mode='train')
# test_set=HandGestureDataset(recognition_frames=recognition_frames, mode='test')

BATCH_SIZE = 10
NUM_WORKERS = 4
dl_train = DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
dl_test = DataLoader(test_set, shuffle=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)



# These are the parameters to be used
onnxfilename='mlp.onnx'
nInput = 78*recognition_frames
nOutput = 4
nLayer = 10
nHidden = 64
act_fn = nn.ReLU()

model = MLP(nInput, nOutput, nLayer, nHidden, act_fn).cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 100
for epoch in range(1, epochs + 1):
    # time.sleep(1)
    train(model, dl_train, optimizer, epoch, log_interval=100)
    test(model, dl_test)

print ('Training is finished.')


# export to ONNX

dummy_input = torch.randn(1, nInput, device="cuda")

input_names = [ "input1" ]
output_names = [ "output1" ]

torch.onnx.export(model, dummy_input, onnxfilename, verbose=True, input_names=input_names, output_names=output_names)
with torch.no_grad():
    torch_output=model(dummy_input)

import onnxruntime as ort

ort_session = ort.InferenceSession(onnxfilename)

outputs = ort_session.run(
    None,     
    {"input1": dummy_input.cpu().numpy().astype(np.float32)},
)
onnx_output=outputs[0]

print(torch_output)
print(onnx_output)
