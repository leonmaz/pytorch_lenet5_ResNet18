import torch
import numpy as np
import torchvision.datasets as datasets
from torchvision import transforms


CIFAR10_ROOT = "./data" 

trainset = datasets.CIFAR10(root=CIFAR10_ROOT, train=True, download=True, transform=transforms.ToTensor())


""" def load_cifar10():

    # only training label is needed for doing split
    #train_label = np.array(trainset.targets)
    return train_label """


def cifar10_mean_std():
    imgs = [item[0] for item in trainset] # item[0] and item[1] are image and its label
    imgs = torch.stack(imgs, dim=0).numpy()

    # calculate mean over each channel (r,g,b)
    mean_r = imgs[:,0,:,:].mean()
    mean_g = imgs[:,1,:,:].mean()
    mean_b = imgs[:,2,:,:].mean()

    # calculate std over each channel (r,g,b)
    std_r = imgs[:,0,:,:].std()
    std_g = imgs[:,1,:,:].std()
    std_b = imgs[:,2,:,:].std()

    return max, mean_r, mean_g, mean_b, std_r, std_g, std_b 

""" def get_site_class_summary(train_label, site_idx):
    class_sum = {}

    for site, data_idx in site_idx.items():
        unq, unq_cnt = np.unique(train_label[data_idx], return_counts=True)
        tmp = {int(unq[i]): int(unq_cnt[i]) for i in range(len(unq))}
        class_sum[site] = tmp
    return class_sum """


def main():
    cifar10_mean_std()
    print(("CIFAR10 DATASET"))
    print(f'Scaled Mean Pixel Value (R G B): {cifar10_mean_std()[1],cifar10_mean_std()[2],cifar10_mean_std()[3]} \nScaled Pixel Value Std (R G B): {cifar10_mean_std()[4],cifar10_mean_std()[5],cifar10_mean_std()[6]}')


if __name__ == "__main__":
    main()