import numpy as np
import torchvision.datasets as datasets


MNIST_ROOT = "./data" 

def load_mnist():
    trainset = datasets.MNIST(root=MNIST_ROOT, train=True, download=True)

    max = trainset.data.max()
    
    mean = trainset.data.float().mean() / max
    std = trainset.data.float().std() / max

    # only training label is needed for doing split
    #train_label = np.array(trainset.targets)
    return trainset, max, mean, std

""" def get_site_class_summary(train_label, site_idx):
    class_sum = {}

    for site, data_idx in site_idx.items():
        unq, unq_cnt = np.unique(train_label[data_idx], return_counts=True)
        tmp = {int(unq[i]): int(unq_cnt[i]) for i in range(len(unq))}
        class_sum[site] = tmp
    return class_sum """


def main():
    load_mnist()
    print(("MNIST DATASET"))
    print(f'Max Pixel Value: {load_mnist()[1]}')
    print(f'Scaled Mean Pixel Value: {load_mnist()[2]} \nScaled Pixel Value Std: {load_mnist()[3]}')


if __name__ == "__main__":
    main()