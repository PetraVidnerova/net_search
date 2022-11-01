import click
import sys 
from operator import mul
from functools import reduce

import numpy as np

from tqdm import tqdm 
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from utils import train, evaluate
from config import read_config
from load_networks import load_network
from data import create_data_loader

PRETRAINED=False
BATCH_SIZE=512

def evaluate_network(net, train_cfg, random_seed=None, device="cpu"):
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    train_dl, val_dl, _ = create_data_loader(BATCH_SIZE, train_val=True, test=False)
        
    net = train(
        net,
        train_dl, val_dl, 
        **train_cfg,
        device=device
    )
    _, _, test_dl = create_data_loader(BATCH_SIZE, train_val=False, test=True)
    evaluate(net, test_dl, device)
    


@click.command()
@click.argument('networks')
@click.argument('traincfg', default="train_cfg_example.yaml")
def test(networks, traincfg):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    train_cfg = read_config(traincfg)

    for name, net in  load_network(networks, input_shape=(3, 227,227), random_seed=7):
        
        print(net)
        evaluate_network(net, train_cfg, random_seed=42, device=device)
        
        torch.save(net, f"{name}.pt")



if __name__ == "__main__":
    test()
