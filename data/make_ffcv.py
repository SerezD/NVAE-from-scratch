import argparse
from typing import Any
from PIL.Image import Image
import numpy as np
import torch

from ffcv import DatasetWriter
from ffcv.fields import Field, RGBImageField, BytesField, IntField, FloatField, NDArrayField, JSONField, \
    TorchTensorField
from torch.utils.data import Dataset
import os

from torchvision.datasets import ImageFolder

from data.datasets import ImageDataset, ImageLabelDataset


def get_args():

    parser = argparse.ArgumentParser(
        description='Define an Image Dataset using ffcv for fast data loading')

    parser.add_argument('--max_resolution', type=int, default=256)
    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument('--has_labels', action='store_true')
    parser.add_argument('--train_folder', type=str, default=None)
    parser.add_argument('--val_folder', type=str, default=None)
    parser.add_argument('--test_folder', type=str, default=None)
    parser.add_argument('--predict_folder', type=str, default=None)

    return parser.parse_args()


def create_beton_wrapper(torch_dataset: Dataset, output_path: str, fields: tuple | None = None,
                         page_size: int | None = None, num_workers: int = -1,
                         indices: list[int] | None = None, chunksize: int = 100, shuffle_indices: bool = False,
                         ) -> None:
    """
    Simple utility function that allows the creation of .beton files from Torch Datasets.
    References: https://docs.ffcv.io/writing_datasets.html

    The dataset can have any number/type of parameters in the __init__ method.

    Constraints on the __get_item__ method:
        According to the official ffcv docs, the dataset must return a tuple object of any length.

        The type of the elements inside the tuple is automatically mapped to the ffcv.fields admitted types:
        https://docs.ffcv.io/api/fields.html

    Default Fields (depending on objects obtained from Dataset.__get_item__):

    PIL Images - RGBImageField(writemode='jpg')

    Integers - IntField()

    Floats - FloatField()

    Numpy arrays - NDArrayField(obj.dtype, obj.shape)

    Dict - JSONField()

    Torch tensor - TorchTensorField(obj.dtype, obj.shape)

    1D uint8 numpy array - BytesField()

    :param torch_dataset: Pytorch Dataset object (https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).
    :param output_path: desired path for .beton output file, "/" separated. E.g. "./my_dataset.beton"
    :param fields: use this if you want to use Fields different from default.
                   The length and order must respect the "get_item" return of the torch Dataset.
                   If you want to overwrite only some fields, pass None to the remaining positions.
    :param page_size: page size internally used. (optional argument of DatasetWriter object)
    :param num_workers: Number of processes to use. (optional argument of DatasetWriter object)
    :param indices: Use a subset of the dataset specified by indices. (optional argument of from_indexed_dataset method)
    :param chunksize: Size of chunks processed by each worker during conversion.
                      (optional argument of from_indexed_dataset method)
    :param shuffle_indices: Shuffle order of the dataset. (optional argument of from_indexed_dataset method)
    """

    def field_to_str(f: Field) -> str:
        mapping = {RGBImageField: "image",
                   BytesField: "bytes",
                   IntField: "int",
                   FloatField: "float",
                   NDArrayField: "array",
                   JSONField: "json",
                   TorchTensorField: "tensor"}
        return mapping[f]

    def obj_to_field(obj: Any) -> Field:

        if isinstance(obj, Image):
            return RGBImageField(write_mode="jpg")

        elif isinstance(obj, int):
            return IntField()

        elif isinstance(obj, float):
            return FloatField()

        elif isinstance(obj, np.ndarray) and not isinstance(obj[0], np.uint8):
            return NDArrayField(obj.dtype, obj.shape)

        elif isinstance(obj, np.ndarray) and isinstance(obj[0], np.uint8):
            return BytesField()

        elif isinstance(obj, dict):
            return JSONField()

        elif isinstance(obj, torch.Tensor):
            return TorchTensorField(obj.dtype, obj.shape)

        else:
            raise AttributeError(f"FFCV dataset can not manage {type(obj)} objects")

    # get default page size (4 * MIN_PAGE_SIZE): --> https://github.com/libffcv/ffcv/blob/main/ffcv/writer.py
    page_size = 4 * (2 ** 21) if page_size is None else page_size

    # 1. format output path
    assert len(output_path) > 0, 'param: output_path cannot be an empty string'

    if not output_path.endswith('.beton'):
        output_path = f'{output_path}.beton'

    # find dir
    dir_name = '/'.join(output_path.split('/')[:-1])
    if not os.path.exists(dir_name):
        print(f'[INFO] Creating output folder: {dir_name}')
        os.makedirs(dir_name)

    # 2. check that dataset __get_item__ returns a tuple and get fields.
    tuple_obj = torch_dataset[0]

    if not isinstance(tuple_obj, tuple):
        raise AttributeError("According to the official ffcv docs, the dataset must return a tuple object. "
                             "See for example: https://docs.ffcv.io/writing_datasets.html")

    if fields is None:
        fields = tuple([None for _ in range(len(tuple_obj))])

    if not len(fields) == len(tuple_obj):
        raise AttributeError("Passed a wrong number of 'fields' objects.\n"
                             f"The __get_item__ method of the specified dataset returns {len(tuple_obj)} elements, but"
                             f" {len(fields)} objects were passed.")

    final_fields = []
    for obj, f in zip(tuple_obj, fields):
        final_fields.append(obj_to_field(obj) if f is None else f)

    # 2. create dict of fields
    final_mapping = {}
    for i, f in enumerate(final_fields):
        final_mapping[f'{field_to_str(type(f))}_{i}'] = f

    # official guidelines: https://docs.ffcv.io/writing_datasets.html
    print(f'[INFO] creating ffcv dataset into file: {output_path}')
    print(f'[INFO] number of items: {len(torch_dataset)}')
    print(f'[INFO] ffcv fields of items: {final_fields}')

    writer = DatasetWriter(output_path, final_mapping, page_size=page_size, num_workers=num_workers)
    writer.from_indexed_dataset(torch_dataset, indices=indices, chunksize=chunksize, shuffle_indices=shuffle_indices)

    print(f'[INFO] Done.')
    print(f'#' * 30)


def main():

    args = get_args()

    dataset_instance = ImageLabelDataset if args.has_labels else ImageDataset

    # https://docs.ffcv.io/api/fields.html
    fields = [RGBImageField(write_mode='jpg', max_resolution=args.max_resolution),]
    if args.has_labels:
        fields.append(IntField())

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    if args.train_folder is not None:

        train_dataset = dataset_instance(folder=args.train_folder, image_size=args.max_resolution, ffcv=True)
        create_beton_wrapper(train_dataset, f"{args.output_folder}/train.beton", fields=fields)

    if args.val_folder is not None:
        val_dataset = dataset_instance(folder=args.val_folder, image_size=args.max_resolution, ffcv=True)
        create_beton_wrapper(val_dataset, f"{args.output_folder}/validation.beton", fields=fields)

    if args.test_folder is not None:
        test_dataset = dataset_instance(folder=args.test_folder, image_size=args.max_resolution, ffcv=True)
        create_beton_wrapper(test_dataset, f"{args.output_folder}/test.beton", fields=fields)

    if args.predict_folder is not None:

        predict_dataset = dataset_instance(folder=args.predict_folder, image_size=args.max_resolution, ffcv=True)
        create_beton_wrapper(predict_dataset, f"{args.output_folder}/predict.beton", fields=fields)


if __name__ == '__main__':

    main()