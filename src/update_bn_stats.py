import argparse
import os

import torch
from torch.cuda.amp import autocast
import torch.multiprocessing as mp
import torch.distributed as dist
from torchvision.utils import make_grid

from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np

from src.hl_autoencoders.NVAE.mine.distributions import DiscMixLogistic
from src.hl_autoencoders.NVAE.mine.model import AutoEncoder


def parse_args():

    parser = argparse.ArgumentParser('update BN statistics for sampling')

    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='NVAE checkpoint path file')

    parser.add_argument('--temperature', type=float, required=True,
                        help='Sampling Temperature')

    args = parser.parse_args()

    # check checkpoint file exists
    if not os.path.exists(f'{args.checkpoint_path}'):
        raise FileNotFoundError(f'could not find the specified checkpoint: {args.checkpoint_path}')

    return args


def setup(rank: int, world_size: int, train_conf: dict):

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # ensures that weight initializations are all the same
    torch.manual_seed(train_conf['seed'])
    np.random.seed(train_conf['seed'])
    torch.cuda.manual_seed(train_conf['seed'])
    torch.cuda.manual_seed_all(train_conf['seed'])


def cleanup():
    dist.destroy_process_group()


def get_model_conf(filepath: str):
    import yaml

    # load params
    with open(filepath, 'r', encoding='utf-8') as stream:
        params = yaml.safe_load(stream)

    return params


def update_bn(rank, args: argparse.Namespace, device: str = 'cuda:0'):

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    config = checkpoint['configuration']

    setup(rank, 1, config['training'])

    # create model and move it to GPU with id rank
    model = AutoEncoder(config['autoencoder'], config['resolution']).to(device)

    print(f'[INFO] Loading checkpoint from: {args.checkpoint_path}')

    model.load_state_dict(checkpoint['state_dict'])

    print(f'[INFO] adjusting batch norm...')
    with torch.no_grad():
        model.train()
        with autocast():
            for _ in tqdm(range(500)):
                model.sample(100, args.temperature, device)
        model.eval()

    # save updated checkpoint
    checkpoint['state_dict_adjusted'] = model.to('cpu').state_dict()
    torch.save(checkpoint, args.checkpoint_path)

    cleanup()


if __name__ == '__main__':

    arguments = parse_args()

    try:
        mp.spawn(update_bn, args=(arguments,), nprocs=1)
    except KeyboardInterrupt as k:
        dist.destroy_process_group()
        raise k
