import argparse
import os

import torch
from torchvision.utils import make_grid

from matplotlib import pyplot as plt
from tqdm import tqdm

from src.model import AutoEncoder


def parse_args():

    parser = argparse.ArgumentParser('NVAE sampling of some images')

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

    args = parser.parse_args()

    # check checkpoint file exists
    if not os.path.exists(f'{args.checkpoint_path}'):
        raise FileNotFoundError(f'could not find the specified checkpoint: {args.checkpoint_path}')

    # create out directory
    args.save_path = f'{args.save_path}/temp={args.temperature}/'
    if not os.path.exists(f'{args.save_path}'):
        os.makedirs(f'{args.save_path}')

    return args


def sample(args: argparse.Namespace, device: str = 'cuda:0'):

    # get params
    bs = args.batch_size
    num_samples = args.n_samples
    temperature = args.temperature

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    config = checkpoint['configuration']

    # create model and move it to device
    model = AutoEncoder(config['autoencoder'], config['resolution'])
    model.load_state_dict(checkpoint[f'state_dict_temp={args.temperature}'])
    model.eval().to(device)

    print(f'[INFO] sampling...')
    with torch.no_grad():

        for n in tqdm(range(0, num_samples, bs)):

            samples = model.sample(bs, temperature, device)
            imgs = make_grid(samples).cpu().numpy().transpose(1, 2, 0)

            plt.imshow(imgs)
            plt.axis(False)
            plt.title(f"Temperature={temperature}")
            plt.savefig(f"{args.save_path}/samples_{n}:{n+bs}.png")
            plt.close()


if __name__ == '__main__':

    sample(parse_args())
