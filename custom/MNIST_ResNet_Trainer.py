import torch
import torchvision
import torchvision.transforms as  transforms

import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np 

from models.networks import ResNet, BasicBlock
from datasets.mnist_load import load_mnist


transform = transforms.Compose(
    [transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(mean = (load_mnist()[1]), std = (load_mnist()[2]))])

batch_size = 4

trainset = torchvision.datasets.MNIST(root='./data', train=True, 
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)



device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")


model = ResNet(1, 18, BasicBlock, 10)
model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, (images, labels) in enumerate(trainloader):  
        # get the inputs; data is a list of [inputs, labels]
        images = images.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')


if not os.path.exists('./saved_models'):
    os.makedirs('./saved_models')
path = './saved_models/MNIST_resnet.pth'
torch.save(model.state_dict(), path)

print('Model Saved')

