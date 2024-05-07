import yaml
import torch

from data.loading_utils import ffcv_loader


def get_model_conf(filepath: str) -> dict:
    """
    :param filepath: .yaml configuration file
    :return: parameters as dictionary
    """

    # load params
    with open(filepath, 'r', encoding='utf-8') as stream:
        params = yaml.safe_load(stream)

    return params


def prepare_data(rank: int, world_size: int, data_path: str, conf: dict):

    image_size = conf['resolution'][1]
    batch_size = conf['training']['cumulative_bs'] // world_size
    seed = int(conf['training']['seed'])
    is_distributed = world_size > 1

    train_dataloader = ffcv_loader(f'{data_path}/train.beton', batch_size, image_size, seed, rank,
                                   is_distributed, is_train=True)
    val_dataloader = ffcv_loader(f'{data_path}/validation.beton', batch_size, image_size, seed, rank,
                                 is_distributed, is_train=False)

    return train_dataloader, val_dataloader


def kl_balancer(kl_unbalanced_terms: torch.Tensor, beta: float = 1.0, balance: bool = False,
                alpha: torch.Tensor = None):

    if balance and beta < 1.0:

        # done only during training warmup phase

        device = kl_unbalanced_terms.device
        alpha = alpha.to(device)  # terms depending on groups

        kl_terms = torch.mean(kl_unbalanced_terms, dim=0)  # mean on batch

        # proportional to kl_terms (on all devices)
        kl_coefficients = torch.mean(torch.abs(kl_unbalanced_terms), dim=0, keepdim=True)

        # set coefficients as summing to num_groups
        kl_coefficients = kl_coefficients / alpha  # divide by spatial resolution (alpha)
        kl_coefficients = kl_coefficients / torch.sum(kl_coefficients, dim=1, keepdim=True)  # normalize -> sum to 1
        kl_coefficients = (kl_coefficients * kl_terms.shape[0]).squeeze(0).detach()  # sum to num_groups
        total_kl = torch.sum(kl_terms * kl_coefficients, dim=0, keepdim=True)

        # for reporting
        kl_gammas = kl_coefficients
        kl_terms = kl_terms.detach()

    else:

        # after warmup and validation
        total_kl = torch.sum(kl_unbalanced_terms, dim=1)            # sum of each component (not balanced)
        kl_terms = torch.mean(kl_unbalanced_terms, dim=0).detach()  # mean on batch
        kl_gammas = torch.ones_like(kl_terms)

    return beta * total_kl, kl_gammas, kl_terms
