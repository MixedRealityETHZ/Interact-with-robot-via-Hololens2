import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms


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

# These are the parameters to be used
onnxfilename='mlp.onnx'
nInput = 1000
nOutput = 10
nLayer = 5
nHidden = 32
act_fn = nn.ReLU()

model = MLP(nInput, nOutput, nLayer, nHidden, act_fn).cuda()

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
exit(0)
dummy_input = torch.randn(10, 3, 224, 224, device="cuda")
model = torchvision.models.alexnet(pretrained=True).cuda()

# Providing input and output names sets the display names for values
# within the model's graph. Setting these does not change the semantics
# of the graph; it is only for readability.
#
# The inputs to the network consist of the flat list of inputs (i.e.
# the values you would pass to the forward() method) followed by the
# flat list of parameters. You can partially specify names, i.e. provide
# a list here shorter than the number of inputs to the model, and we will
# only set that subset of names, starting from the beginning.
input_names = [ "actual_input_1" ] #+ [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)

import onnxruntime as ort

ort_session = ort.InferenceSession("alexnet.onnx")

outputs = ort_session.run(
    None,
    {"actual_input_1": np.random.randn(10, 3, 224, 224).astype(np.float32)},
)
print(outputs[0].shape)
# print(outputs[0])