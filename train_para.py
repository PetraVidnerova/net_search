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
from torch.multiprocessing import set_start_method

from utils import train, evaluate
from config import read_config
from load_networks import load_network_configs, create_network
from data import create_data_loader
from pool import Pool

PRETRAINED=False
BATCH_SIZE=256

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
    

def evaluate_task(task_config, device):
    netstr, train_cfg, input_shape, random_seed = task_config
    try:
        name, net =  create_network(netstr, input_shape, random_seed=random_seed)
        evaluate_network(net, train_cfg, random_seed=random_seed, device=device)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"Network {name} out of memory.")
            return
        else:
            print(f"Network {name}: ", e)
            return
                  
    torch.save(net, f"{name}.pt")
    
    
@click.command()
@click.argument('networks')
@click.argument('traincfg', default="train_cfg_example.yaml")
def test(networks, traincfg):
    
    train_cfg = read_config(traincfg)
    input_shape = (3, 227, 227)
    
    if torch.cuda.is_available():
        processors = torch.cuda.device_count()
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        devices = ["cpu"] * 2
        processors = 2
    
    pool = Pool(processors=processors, evalfunc=evaluate_task, devices=devices)

    num = 0
    for cfg in load_network_configs(networks):
        
        task_config = (cfg, train_cfg, input_shape, 42)
        pool.putQuerry(task_config)
        num += 1

    for _ in range(num):
        pool.getAnswer()

    pool.close()
    print("FINISHED")
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    # train_cfg = read_config(traincfg)

    # for name, net in  load_network(networks, input_shape=(3, 227,227), random_seed=7):
        
    #     print(net)
    #     evaluate_network(net, train_cfg, random_seed=42, device=device)
        
    #     torch.save(net, f"{name}.pt")



if __name__ == "__main__":
    set_start_method('spawn')
    test()
