"""
check if reconstruction accuracy drops (is it an attack already?).
"""
import os

import numpy as np
import torch
from einops import pack
from torch.cuda.amp import autocast
from torchvision.utils import make_grid

from robustbench import load_cifar10

from matplotlib import pyplot as plt
from tqdm import tqdm

from src.hl_autoencoders.NVAE.mine.distributions import DiscMixLogistic
from src.hl_autoencoders.NVAE.mine.model import AutoEncoder as NVAE
from src.hl_autoencoders.NVAE.original.model import AutoEncoder
from src.hl_autoencoders.NVAE.original.utils import get_arch_cells
import torch.multiprocessing as mp
import torch.distributed as dist

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision.transforms import ConvertImageDtype


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


def main(rank, temperature: float, IS_OURS: bool, CKPT_NVAE: str, DATA_PATH: str):

    setup(rank=rank, world_size=1, train_conf={'seed': 0})

    if IS_OURS:

        checkpoint = torch.load(CKPT_NVAE, map_location='cpu')

        if 'configuration' in checkpoint.keys():
            config = checkpoint['configuration']
        else:
            conf_file = '/'.join(CKPT_NVAE.split('/')[:-1]) + '/conf.yaml'
            config = get_model_conf(conf_file)

        # create model and move it to GPU with id rank
        nvae = NVAE(config['autoencoder'], config['resolution'])

        nvae.load_state_dict(checkpoint['state_dict_adjusted'])
        # nvae.load_state_dict(checkpoint['state_dict'])
        nvae.to(f'cuda:{rank}').eval()

    else:

        # load nvae pretrained cifar10
        checkpoint = torch.load(CKPT_NVAE, map_location='cpu')

        # get and update args
        args = checkpoint['args']
        args.num_mixture_dec = 10
        args.batch_size = 100

        # init model and load
        arch_instance = get_arch_cells(args.arch_instance)
        nvae = AutoEncoder(args, arch_instance)
        nvae.load_state_dict(checkpoint['state_dict'], strict=False)
        nvae = nvae.cuda().eval()

    # load cifar10
    x_test, y_test = load_cifar10(data_dir=DATA_PATH)
    data_n = x_test.shape[0]
    batch_size = 100

    with torch.no_grad():

        # # TEST L2 ERROR
        # print(f'[INFO] reconstructing...')
        # recons = torch.empty((0, 3, 32, 32), device=f'cuda:{rank}')
        #
        # for b in tqdm(range(0, data_n, batch_size)):
        #
        #     batch_x = x_test[b:b + batch_size].to(f'cuda:{rank}')
        #
        #     # reconstruct cifar10 test set
        #     if IS_OURS:
        #         logits = nvae.autoencode(batch_x, deterministic=True)
        #         batch_recons = DiscMixLogistic(logits, img_channels=3, num_bits=8).mean()
        #     else:
        #         logits = nvae.decode(nvae.encode_deterministic(batch_x))
        #         decoder = nvae.decoder_output(logits)
        #         batch_recons = decoder.mean()
        #
        #     recons, _ = pack([recons, batch_recons], '* c h w')
        #
        # # L2 error:
        # l2 = torch.mean(torch.cdist(x_test.cuda().view(data_n, -1), recons.view(data_n, -1), p=2).diag())
        # print(f"L2 Error: {l2:.5f}")
        #
        # print(f'[INFO] adjusting batch norm...')
        # nvae.train()
        # with autocast():
        #     for _ in tqdm(range(500)):
        #         if IS_OURS:
        #             nvae.sample(batch_size, temperature, f'cuda:{rank}')
        #         else:
        #             nvae.sample(batch_size, temperature)
        #     nvae.eval()

        # SAMPLING
        print(f'[INFO] sampling...')

        # metrics for testing
        fid_score = FrechetInceptionDistance().to(f'cuda:{rank}')
        is_score = InceptionScore().to(f'cuda:{rank}')
        conv = ConvertImageDtype(torch.uint8)

        for b in tqdm(range(0, data_n, batch_size)):

            batch_x = x_test[b:b + batch_size].to(f'cuda:{rank}')

            if IS_OURS:
                with autocast():
                    logits = nvae.sample(batch_size, temperature, f'cuda:{rank}')
                    samples = DiscMixLogistic(logits, img_channels=3, num_bits=8).sample()
            else:
                with autocast():
                    logits = nvae.sample(batch_size, temperature)
                samples = nvae.decoder_output(logits).sample()

            if b == 0:
                imgs = make_grid(samples[:32]).cpu().numpy().transpose(1, 2, 0)
                plt.imshow(imgs)
                plt.axis(False)
                plt.title(f"Temperature={temperature}, OURS: {IS_OURS}")
                plt.savefig(f"./samples_test/samples_{'OURS' if IS_OURS else 'ORIGINAL'}_temp={temperature}.png")
                plt.close()

            # FID
            fid_score.update(conv(samples), real=False)
            fid_score.update(conv(batch_x), real=True)

            is_score.update(conv(samples))

        is_val, is_err = is_score.compute()
        print(f'[INFO] FID score: {fid_score.compute().item():.4f}')
        print(f'[INFO] IS score: {is_val.item():.4f} +- {is_err.item():.4f}')

    cleanup()


if __name__ == '__main__':

    """
    [3SCALES_1GROUP THEIRS (OLD)]
    L2 Error:   0.39227
    FID Score:  68.3391
    IS Score:   4.4885 +- 0.0841 
    
    [3SCALES_1GROUP THEIRS (WANDB LOGGED)]
    L2 Error:   0.35781
    FID Score:  70.8052
    IS Score:   4.3772 +- 0.1242
    
    [3SCALES_1GROUP THEIRS LARGE]
    L2 Error:   0.44198
    FID Score:  63.0407
    IS Score:   4.7530 +- 0.1479
    
    *****************************************
    
    [3SCALES_4GROUPS REFACTORED EPOCH=550]
    L2 Error:   0.66466
    TEMP 0.9 + BN
        FID Score:  57.7157
        IS Score:   5.3854 +- 0.1261
    
    *****************************************
    
    [3SCALES_4GROUPS REGULARIZED EPOCH=240]
    L2 Error:   0.65472
    TEMP 0.9 + BN
        FID Score:  55.3527
        IS Score:   5.5839 +- 0.1364

    *****************************************
    
    [3SCALES_4GROUPS LARGE EPOCH=350]
    L2 Error + BN:  1.32193
    TEMP 1.0 + BN   
        FID score:  54.2549
        IS score:   5.6653 +- 0.1226
        
    *****************************************

    [3SCALES_8GROUPS EPOCH=599]
    L2 Error + BN (T=1.):  0.68003
    L2 Error + BN (T=.6):  0.85704
    
    TEMP 1.0 + BN   
        FID score:  49.8140
        IS score:   5.8071 +- 0.1702
    TEMP 0.6 + BN   
        FID score:  35.9569
        IS score:   7.1258 +- 0.2009
    """

    DATA_PATH = '/media/dserez/datasets/cifar10/'

    IS_OURS = True
    TEMP = 0.6

    if IS_OURS:
        CKPT_NVAE = f'/media/dserez/runs/NVAE/cifar10/ours/8x3/epoch=599.pt'

    else:
        CKPT_NVAE = '/media/dserez/runs/NVAE/cifar10/best/last_hope.pt'

    try:
        mp.spawn(main, args=(TEMP, IS_OURS, CKPT_NVAE, DATA_PATH), nprocs=1)
    except KeyboardInterrupt as k:
        dist.destroy_process_group()
        raise k
