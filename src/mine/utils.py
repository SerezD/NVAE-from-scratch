import torch


def kl_balancer(kl_unbalanced_terms: torch.Tensor, beta: float = 1.0, balance: bool = False, alpha: torch.Tensor = None):

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
