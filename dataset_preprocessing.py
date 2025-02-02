import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import TensorDataset, DataLoader

def load_california_housing_data(test_size=0.2, batch_size=64):
    """
    Loads California Housing dataset, splits into train/test,
    normalizes features, and returns PyTorch DataLoaders.
    """
    # 1. Fetch the dataset
    dataset = fetch_california_housing()
    X, y = dataset.data, dataset.target  # Numpy arrays
    
    # 2. Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # 3. Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 4. Convert to torch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test, dtype=torch.float32).view(-1,1)

    # 5. Create TensorDataset and DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset  = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


import torchvision
import torchvision.transforms as T

def load_cifar10_data(batch_size=64):
    """
    Loads CIFAR-10 dataset via torchvision, applies basic transforms,
    and returns train/test DataLoaders.
    """
    # 1. Define transforms: convert to Tensor, normalize
    transform = T.Compose([
        T.RandomHorizontalFlip(), 
        T.RandomCrop(32, padding=4),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010))
    ])

    # 2. Load train/test sets
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, 
        download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, 
        download=True, transform=transform
    )

    # 3. Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def main():
    # 1. Test California Housing
    cal_train_loader, cal_test_loader = load_california_housing_data()
    print("California Housing: Train batches:", len(cal_train_loader), "Test batches:", len(cal_test_loader))

    # 2. Test CIFAR-10
    cifar_train_loader, cifar_test_loader = load_cifar10_data()
    print("CIFAR-10: Train batches:", len(cifar_train_loader), "Test batches:", len(cifar_test_loader))

if __name__ == "__main__":
    main()

