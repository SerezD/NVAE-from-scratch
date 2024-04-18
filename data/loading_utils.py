from ffcv.fields.basics import IntDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder

import torch

from data.ffcv_augmentations import DivideImage255


def ffcv_loader(data_path: str, batch_size: int, image_size: int, seed: int, rank: int, distributed: bool,
                workers: int = 8, mode: str = 'train', dtype: torch.dtype = torch.float16):

    pipeline = [
        CenterCropRGBImageDecoder((image_size, image_size), ratio=1.),
        ToTensor(),
        ToDevice(torch.device(rank), non_blocking=True),
        ToTorchImage(),
        DivideImage255(dtype=dtype)
    ]

    if mode == 'train':

        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM

        loader = Loader(f'{data_path}/train.beton',
                        batch_size=batch_size,
                        num_workers=workers,
                        order=order,
                        os_cache=True,
                        drop_last=True,
                        seed=seed,
                        pipelines={
                            'image_0': pipeline,
                        },
                        distributed=distributed)
    else:

        order = OrderOption.SEQUENTIAL

        loader = Loader(f'{data_path}/validation.beton',
                        batch_size=batch_size,
                        num_workers=workers,
                        order=order,
                        os_cache=True,
                        drop_last=False,
                        seed=seed,
                        pipelines={
                            'image_0': pipeline,
                        },
                        distributed=distributed)

    return loader


def ffcv_labels_loader(data_path: str, batch_size: int, image_size: int, seed: int, rank: int, distributed: bool,
                       workers: int = 8, mode: str = 'train', dtype: torch.dtype = torch.float32):

    pipeline_imgs = [
        CenterCropRGBImageDecoder((image_size, image_size), ratio=1.),
        ToTensor(),
        ToDevice(torch.device(rank), non_blocking=True),
        ToTorchImage(),
        DivideImage255(dtype=dtype)
    ]

    pipeline_lbls = [
        IntDecoder(),
        ToTensor(),
        ToDevice(torch.device(rank), non_blocking=True)
    ]

    if mode == 'train':

        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM

        loader = Loader(f'{data_path}/train.beton',
                        batch_size=batch_size,
                        num_workers=workers,
                        order=order,
                        os_cache=True,
                        drop_last=True,
                        seed=seed,
                        pipelines={
                            'image_0': pipeline_imgs,
                            'int_1': pipeline_lbls
                        },
                        distributed=distributed)
    else:

        order = OrderOption.SEQUENTIAL

        loader = Loader(f'{data_path}/validation.beton',
                        batch_size=batch_size,
                        num_workers=workers,
                        order=order,
                        os_cache=True,
                        drop_last=False,
                        seed=seed,
                        pipelines={
                            'image_0': pipeline_imgs,
                            'int_1': pipeline_lbls
                        },
                        distributed=distributed)

    return loader
