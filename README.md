############################### Federated Learning Project ###############################

## (Optional) 1. Set up a virtual environment

```
python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv
```
(If needed) make all shell scripts executable using
```
find . -name ".sh" -exec chmod +x {} \;
```
initialize virtual environment.
```
source ./virtualenv/set_env.sh
```


## 2. Centralized Machine Learning 

I) MNIST DATASET

a. Simple CNN model (based on LeNet-5)

```
./MNIST_SimpleCNN.sh
```
b. ResNet-18 Model

```
./MNIST_ResNet.sh
```

II) CIFAR10 DATASET

a. Simple CNN model (based on LeNet-5)

```
./CIFAR10_SimpleCNN.sh
```
b. ResNet-18 Model

```
./CIFAR10_ResNet.sh
```

III) CIFAR100 DATASET

a. Simple CNN model (based on LeNet-5)

```
./CIFAR100_SimpleCNN.sh
```
b. ResNet-18 Model

```
./CIFAR100_ResNet.sh
```
