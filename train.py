import click
import sys 
from operator import mul
from functools import reduce

from tqdm import tqdm 
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from utils import train, evaluate
from config import read_config

PRETRAINED=False
BATCH_SIZE=512

@click.command()
@click.argument('filename', default="train_cfg_example.yaml")
def test(filename):
    net = torchvision.models.alexnet(pretrained=PRETRAINED)
    net.classifier[4] = nn.Linear(4096,1024)
    net.classifier[6] = nn.Linear(1024,10)
    #net.classifier.append(nn.Softmax(1)) softmax is part of entropy loss
    print(net)

    
    transform_train = transforms.Compose([
        transforms.Resize((227,227)),
        transforms.RandomHorizontalFlip(p=0.7),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    torch.manual_seed(42)
    train_ds = CIFAR10("./data/", train=True, download=True, transform=transform_train) #40,000 original images + transforms

    #BATCH_SIZE=512
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [45000, 5000])
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

    train_cfg = read_config(filename)

    net = train(
        net,
        train_dl, val_dl, 
        **train_cfg,
        device=device
    )
    torch.save(net, f"alexnet_{'pretrained' if PRETRAINED else 'from_scratch'}_{BATCH_SIZE}.pt")


    test_ds = CIFAR10("../example/data/", train=False, download=True,
                      transform=transform_train)
    test_dl = DataLoader(test_ds, batch_size=64, num_workers=8, shuffle=False)

    evaluate(net, test_dl, device)


if __name__ == "__main__":
    test()
