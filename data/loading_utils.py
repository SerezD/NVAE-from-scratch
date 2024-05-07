from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, RandomHorizontalFlip
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder
import torch

from data.ffcv_augmentations import DivideImage255


def ffcv_loader(data_path: str, batch_size: int, image_size: int, seed: int, rank: int, distributed: bool,
                is_train: bool, workers: int = 8, dtype: torch.dtype = torch.float16,
                image_name_in_ppl: str = 'image_0'):
    """
    :param data_path: path to .beton file (can be created with script make_ffcv.py)
    :param batch_size:
    :param image_size:
    :param seed: random seed to ensure reproducibility
    :param rank: local rank to use to return batch
    :param distributed:
    :param is_train: if True, applies extra transforms for Data Augmentation
    :param workers: can highly affect loading speed. Try different values depending on architecture.
    :param dtype:
    :param image_name_in_ppl: name of image in .beton file for loading. If using provided script (make_ffcv.py) to
    create the beton files, can leave default "image_0".
    """
    pipeline = [
        CenterCropRGBImageDecoder((image_size, image_size), ratio=1.),
        ToTensor(),
        ToTorchImage(channels_last=False),
        ToDevice(torch.device(rank), non_blocking=True),
        DivideImage255(dtype=dtype)
    ]

    if is_train:

        # add random data augmentations
        pipeline = [pipeline[0], RandomHorizontalFlip()] + pipeline[1:]

        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM

        loader = Loader(f'{data_path}/train.beton',
                        batch_size=batch_size,
                        num_workers=workers,
                        order=order,
                        os_cache=True,
                        drop_last=True,
                        seed=seed,
                        pipelines={
                            f'{image_name_in_ppl}': pipeline,
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
                            f'{image_name_in_ppl}': pipeline,
                        },
                        distributed=distributed)

    return loader
