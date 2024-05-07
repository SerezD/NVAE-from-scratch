import argparse
import os

import torch
from torch.cuda.amp import autocast
from torchvision.utils import make_grid


from matplotlib import pyplot as plt
from tqdm import tqdm

from data.loading_utils import ffcv_loader
from src.model import AutoEncoder as NVAE

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics import MeanSquaredError
from torchvision.transforms import ConvertImageDtype


def parse_args():

    parser = argparse.ArgumentParser('compute L2 reconstructions, FID and IS scores')

    parser.add_argument('--images_path', type=str, required=True,
                        help='.beton file with ground truth images')

    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='NVAE checkpoint path file')

    parser.add_argument('--temperature', type=float, required=True,
                        help='Sampling Temperature')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size for sampling and reconstructions')

    args = parser.parse_args()

    # check images exists
    if not os.path.exists(f'{args.images_path}'):
        raise FileNotFoundError(f'could not find the specified .beton file: {args.images_path}')

    # check checkpoint file exists
    if not os.path.exists(f'{args.checkpoint_path}'):
        raise FileNotFoundError(f'could not find the specified checkpoint: {args.checkpoint_path}')

    return args


def main(arguments: argparse.Namespace):

    checkpoint = torch.load(arguments.checkpoint_path, map_location='cpu')
    config = checkpoint['configuration']

    # create model and move it to GPU with id rank
    nvae = NVAE(config['autoencoder'], config['resolution'])

    nvae.load_state_dict(checkpoint[f'state_dict_temp={arguments.temperature}'])
    nvae.to(f'cuda:0').eval()

    # load data
    image_size = config['resolution'][1]

    dataloader = ffcv_loader(arguments.images_path, arguments.batch_size, image_size, seed=0, rank=0,
                             distributed=False, is_train=False, dtype=torch.float32)

    # metrics for testing
    l2_score = MeanSquaredError().to(f'cuda:0')
    fid_score = FrechetInceptionDistance().to(f'cuda:0')
    is_score = InceptionScore().to(f'cuda:0')
    conv = ConvertImageDtype(torch.uint8)

    print('[INFO] Starting evaluations')

    with torch.no_grad():

        for (batch, ) in tqdm(dataloader):

            reconstructions = nvae.reconstruct(batch, deterministic=True)
            samples = nvae.sample(num_samples=arguments.batch_size, temperature=arguments.temperature, device='cuda:0')

            # L2
            l2_score.update(reconstructions, batch)

            # FID
            fid_score.update(conv(samples), real=False)
            fid_score.update(conv(batch), real=True)

            # IS SCORE
            is_score.update(conv(samples))

        # print out
        print(f'[INFO] Finished evaluations for NVAE with temperature {arguments.temperature:.2f}')

        print(f'[INFO] L2 on Reconstructions: {l2_score.compute().item():.8f}')

        print(f'[INFO] FID score: {fid_score.compute().item():.4f}')

        is_val, is_err = is_score.compute()
        print(f'[INFO] IS score: {is_val.item():.4f} +- {is_err.item():.4f}')


if __name__ == '__main__':

    main(parse_args())
