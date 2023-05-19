import torch
import torchvision
import torchvision.transforms as  transforms

import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np 
import os 

from models.networks import ResNet, BasicBlock
from datasets.mnist_load import load_mnist

path = './saved_models/MNIST_resnet.pth'
if not os.path.exists(path):
    print("Model is not created, run file 'MNIST_ResNet_Trainer.py' first")
    exit()

model = ResNet(1, 18, BasicBlock, 10)
model.load_state_dict(torch.load(path))


transform = transforms.Compose(
    [transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(mean = (load_mnist()[1]), std = (load_mnist()[2]))])

batch_size = 1000

testset = torchvision.datasets.MNIST(root='./data', train=False,
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


examples = enumerate(testloader)
batch_idx, (example_data, example_targets) = next(examples)


with torch.no_grad():
  output = model(example_data)


plt.figure()
for i in range(10):
  plt.subplot(2,5,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Prediction: {}".format(
    output.data.max(1, keepdim=True)[1][i].item()))
  plt.xticks([])
  plt.yticks([])
plt.show()
