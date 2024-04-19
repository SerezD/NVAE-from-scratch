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

    parser = argparse.ArgumentParser('NVAE sampling some images')

    parser.add_argument('--conf_file', type=str, required=True,
                        help='.yaml configuration file')

    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='NVAE checkpoint path file')

    parser.add_argument('--save_path', type=str, default='./samples/',
                        help='where to save sampled images (directory path)')

    parser.add_argument('--n_samples', type=int, default=16,
                        help='number of samples to generate in total')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='number of samples to generate at each step')

    parser.add_argument('--temperature', type=float, default=1.,
                        help='temperature parameter for sampling')

    parser.add_argument('--adjust_bn', action='store_true', default=False,
                        help='weather to recompute batch norm stats before sampling')

    args = parser.parse_args()

    # check conf file exists
    if not os.path.exists(f'{args.conf_file}'):
        raise FileNotFoundError(f'could not find the specified configuration file: {args.conf_file}')

    # check checkpoint file exists
    if not os.path.exists(f'{args.checkpoint_path}'):
        raise FileNotFoundError(f'could not find the specified checkpoint: {args.checkpoint_path}')

    # create out directory
    args.save_path = f'{args.save_path}/temp={args.temperature}_bn_adjusted={args.adjust_bn}/'
    if not os.path.exists(f'{args.save_path}'):
        os.makedirs(f'{args.save_path}')

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


def sample(rank, args: argparse.Namespace, device: str = 'cuda:0'):

    # get params
    bs = args.batch_size
    num_samples = args.n_samples
    temperature = args.temperature
    config = get_model_conf(args.conf_file)

    setup(rank, 1, config['training'])

    # create model and move it to GPU with id rank
    model = AutoEncoder(config['autoencoder'], config['resolution']).to(device)

    print(f'[INFO] Loading checkpoint from: {args.checkpoint_path}')

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    with torch.no_grad():
        if args.adjust_bn:
            print(f'[INFO] adjusting batch norm...')
            model.train()
            with autocast():
                for _ in tqdm(range(500)):
                    model.sample(bs, temperature, device)
            model.eval()

        # sample
        print(f'[INFO] sampling...')

        for n in tqdm(range(0, num_samples, bs)):

            logits = model.sample(bs, temperature, device)
            samples = DiscMixLogistic(logits, img_channels=3, num_bits=8).sample()
            imgs = make_grid(samples).cpu().numpy().transpose(1, 2, 0)

            plt.imshow(imgs)
            plt.axis(False)
            plt.title(f"Temperature={temperature}, BN adjusted: {args.adjust_bn}")
            plt.savefig(f"{args.save_path}/samples_{n}:{n+bs}.png")
            plt.close()

    cleanup()


if __name__ == '__main__':

    arguments = parse_args()

    try:
        mp.spawn(sample, args=(arguments,), nprocs=1)
    except KeyboardInterrupt as k:
        dist.destroy_process_group()
        raise k
