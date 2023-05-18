import numpy as np
import torchvision.datasets as datasets


CIFAR100_ROOT = "../data" 

def load_cifar10():
    train_dataset = datasets.CIFAR100(root=CIFAR100_ROOT, train=True, download=True)

    # only training label is needed for doing split
    train_label = np.array(train_dataset.targets)
    return train_label


def get_site_class_summary(train_label, site_idx):
    class_sum = {}

    for site, data_idx in site_idx.items():
        unq, unq_cnt = np.unique(train_label[data_idx], return_counts=True)
        tmp = {int(unq[i]): int(unq_cnt[i]) for i in range(len(unq))}
        class_sum[site] = tmp
    return class_sum


def main():
    load_cifar100()


if __name__ == "__main__":
    main()