import torch
import torchvision
import torchvision.transforms as  transforms

import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np 
import os 

from networks import SimpleCNN

path = 'models/cifar10_simple.pth'
if not os.path.exists(path):
    print("Model is not created, run file 'SimpleCNNtrainer_central.py' first")
    exit()

model = SimpleCNN(3,10)
model.load_state_dict(torch.load(path))


transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)


correct = 0
total = 0

for data in testloader:
    inputs, labels = data
    # calculate outputs by running images through the network
    outputs = model(inputs)
    # the class with the highest energy is what we choose as prediction
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
