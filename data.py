import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader


def create_data_loader(batch_size, train_val=True, test=True):

    transform_train = transforms.Compose([
        transforms.Resize((227,227)),
        transforms.RandomHorizontalFlip(p=0.7),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if train_val:
        train_ds = CIFAR10("./data/", train=True, download=True, transform=transform_train) #40,000 original images + transforms

        train_ds, val_ds = torch.utils.data.random_split(train_ds, [45000, 5000])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8)
    else:
        train_dl, val_dl = None, None

    if test:
        test_ds = CIFAR10("./data/", train=False, download=True,
                          transform=transform_train)
        test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=8, shuffle=False)
    else:
        test_dl = None

    return train_dl, val_dl, test_dl
