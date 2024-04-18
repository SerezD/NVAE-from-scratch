import argparse
import os
import warnings
import socket
from typing import Any
import wandb

import numpy as np

import torch
from einops import pack
from scheduling_utils.schedulers_cpp import LinearCosineScheduler, CosineScheduler
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from pytorch_model_summary import summary
from torchvision.utils import make_grid
from kornia.augmentation import AugmentationSequential, RandomHorizontalFlip
from tqdm import tqdm

from data.loading_utils import ffcv_loader
from src.hl_autoencoders.NVAE.mine.distributions import DiscMixLogistic
from src.hl_autoencoders.NVAE.mine.model import AutoEncoder
from src.hl_autoencoders.NVAE.mine.utils import kl_balancer


def parse_args():

    parser = argparse.ArgumentParser('NVAE training')

    parser.add_argument('--run_name', type=str, required=True,
                        help='unique name of the training run')

    parser.add_argument('--conf_file', type=str, required=True,
                        help='.yaml configuration file')

    parser.add_argument('--data_path', type=str, required=True,
                        help='directory of ffcv datasets (.beton files)')

    parser.add_argument('--checkpoint_base_path', type=str, default='./runs/',
                        help='directory where checkpoints are saved')

    parser.add_argument('--resume_from', type=str, default=None,
                        help='if specified, resume training from this checkpoint')

    parser.add_argument('--logging', help='if passed, wandb logger is used', action='store_true')

    parser.add_argument('--wandb_project', type=str, help='project name for wandb logger', default='nvae')

    parser.add_argument('--wandb_id', type=str,
                        help='wandb id of the run. Useful for resuming logging of a model', default=None)

    args = parser.parse_args()

    if WORLD_RANK == 0:

        # check data dir exists
        if (not os.path.exists(f'{args.data_path}/train.beton') or
                not os.path.exists(f'{args.data_path}/validation.beton')):
            raise FileNotFoundError(f'{args.data_path}/train.beton or {args.data_path}/validation.beton does not exist')

        # check checkpoint file exists
        if args.resume_from is not None and not os.path.exists(f'{args.resume_from}'):
            raise FileNotFoundError(f'could not find the specified checkpoint: {args.resume_from}')

        # create checkpoint out directory
        if not os.path.exists(f'{args.checkpoint_base_path}/{args.run_name}'):
            os.makedirs(f'{args.checkpoint_base_path}/{args.run_name}')

    return args


def get_model_conf(filepath: str):
    import yaml

    # load params
    with open(filepath, 'r', encoding='utf-8') as stream:
        params = yaml.safe_load(stream)

    return params


def setup(train_conf: dict):

    if 'MASTER_ADDR' not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
        if WORLD_RANK == 0:
            warnings.warn("ENV VARIABLE 'MASTER_ADDR' not specified. Setting 'MASTER_ADDR'='localhost'")

    if 'MASTER_PORT' not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
        if WORLD_RANK == 0:
            warnings.warn("ENV VARIABLE 'MASTER_PORT' not specified. 'MASTER_PORT'='29500'")

    # initialize the process group
    # Note: passing rank and work_size to maintain compatibility with non torch run scripts
    dist.init_process_group("nccl", rank=WORLD_RANK, world_size=WORLD_SIZE)

    # ensures that weight initializations are all the same
    torch.manual_seed(train_conf['seed'])
    np.random.seed(train_conf['seed'])
    torch.cuda.manual_seed(train_conf['seed'])
    torch.cuda.manual_seed_all(train_conf['seed'])

    dist.barrier()


def cleanup():
    dist.destroy_process_group()


def prepare_data(rank: int, world_size: int, data_dir: str, conf: dict):

    image_size = conf['resolution'][1]
    batch_size = conf['training']['cumulative_bs'] // world_size
    seed = int(conf['training']['seed'])
    is_distributed = world_size > 1

    train_dataloader = ffcv_loader(data_dir, batch_size, image_size, seed, rank, is_distributed)
    val_dataloader = ffcv_loader(data_dir, batch_size, image_size, seed, rank, is_distributed, mode='validation')

    train_augmentations = AugmentationSequential(RandomHorizontalFlip(p=0.5),
                                                 same_on_batch=False)

    if WORLD_RANK == 0:
        print(f"[INFO] final batch size per device: {batch_size}")

    return train_dataloader, val_dataloader, train_augmentations


def epoch_train(dataloader: DataLoader, augmentations: AugmentationSequential,
                model: AutoEncoder, optimizer: torch.optim.Optimizer, scheduler: Any,
                grad_scaler: GradScaler, kl_params: dict, sr_params: dict,
                total_training_steps: int, global_step: int, run: wandb.run = None):
    """
    :param dataloader: train dataloader.
    :param augmentations: augmentation module.
    :param model: model in training mode. Remember to pass ".module" with DDP.
    :param optimizer: optimizer object from torch.optim.Optimizer.
    :param scheduler: scheduler object from scheduling utils.
    :param grad_scaler: for AMP mode.
    :param kl_params: parameters for kl annealing coefficients.
    :param sr_params: spectral regularization parameters for lambda computation
    :param total_training_steps: total training steps on all epochs.
    :param global_step: used by scheduler.
    :param run: wandb run object (if rank is 0).
    """

    # for logging: loss, rec, kl, spectral, bn
    epoch_losses = torch.empty((0, 5), device=f"cuda:{LOCAL_RANK}")

    for step, x in enumerate(tqdm(dataloader)):

        x = augmentations(x[0])

        # scheduler step
        lr = scheduler.step(global_step)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # use autocast for step (AMP)
        optimizer.zero_grad()
        with autocast():

            # forward pass
            logits, kl_terms = model(x)
            reconstructions = DiscMixLogistic(logits, img_channels=3, num_bits=8).log_prob(x)

            # reconstruction loss
            rec_loss = - torch.sum(reconstructions, dim=1)

            # compute kl weight (Appendix A of NVAE paper, linear warmup of KL Term β for kl_anneal_portion of steps)
            beta = (global_step - float(kl_params["kl_const_portion"]) * total_training_steps)
            beta /= (float(kl_params["kl_anneal_portion"]) * total_training_steps)
            beta = max(min(1.0, beta), float(kl_params["kl_const_coeff"]))

            # balance kl (Appendix A of NVAE paper, γ term on each scale)
            final_kl, kl_gammas, kl_terms = kl_balancer(kl_terms, beta, balance=True, alpha=model.kl_alpha)

            # compute final loss
            loss = torch.mean(rec_loss + final_kl)

            # get spectral regularization coefficient λ
            # from Appendix A (annealing λ):
            # The coefficient of smoothness λ is set to a fixed value {10e-2, 10e-1} for almost all experiments, using
            # 10e-1 only when 10e-2 was unstable.
            # Exception is from CelebA and FFHQ where they anneal it for the same number of iterations as β
            if sr_params["weight_decay_norm_anneal"]:
                lambda_coeff = beta * np.log(float(sr_params["weight_decay_norm"]))
                lambda_coeff += (1. - beta) * np.log(float(sr_params["weight_decay_norm_init"]))
                lambda_coeff = np.exp(lambda_coeff)
            else:
                lambda_coeff = float(sr_params["weight_decay_norm"])

            # add spectral regularization and batch norm regularization terms to loss
            spectral_norm_term = model.compute_sr_loss()
            batch_norm_term = model.batch_norm_loss()
            loss += lambda_coeff * spectral_norm_term + lambda_coeff * batch_norm_term

        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()

        # TODO TEST
        # flag = False
        # count = 0
        # for n, p in model.named_parameters():
        #     a = p.detach().clone()
        #     dist.all_reduce(p.data, op=dist.ReduceOp.SUM)
        #     p.data /= WORLD_SIZE
        #     if (a.data - p.data).sum().item() > 0:
        #         flag = True
        #         count += 1
        #         if count == 1:
        #             print(f"AFTER BACKWARD: {n} is different across ranks!")
        #             print(f"SUM OF DIFFERENCES: {(a.data - p.data).sum().item()}")
        # if flag:
        #     raise RuntimeError(f"AFTER BACKWARD: {count} parameters diverge!")

        # to log at each step
        dist.all_reduce(kl_gammas, op=dist.ReduceOp.SUM)
        dist.all_reduce(kl_terms, op=dist.ReduceOp.SUM)
        kl_gammas = kl_gammas / WORLD_SIZE
        kl_terms = kl_terms / WORLD_SIZE

        if WORLD_RANK == 0:

            log_dict = {'lr': lr, 'KL beta': beta, 'Lambda': lambda_coeff}
            for i, (k, v) in enumerate(zip(kl_gammas.cpu().numpy(), kl_terms.cpu().numpy())):
                log_dict[f'KL gamma {i}'] = k
                log_dict[f'KL term {i}'] = v

            run.log(log_dict, step=global_step)

        # save all loss terms
        losses = torch.stack(
            [loss, rec_loss.mean(), final_kl.mean(), spectral_norm_term * lambda_coeff, batch_norm_term * lambda_coeff],
            dim=0).detach()

        dist.all_reduce(losses, op=dist.ReduceOp.SUM)
        losses = losses / WORLD_SIZE

        # loss, rec, kl, spectral, bn
        epoch_losses, _ = pack([epoch_losses, losses], '* n')

        global_step += 1

    # log epoch loss
    if WORLD_RANK == 0:

        epoch_losses = torch.mean(epoch_losses, dim=0)

        run.log(
            {
                "train/loss": epoch_losses[0].item(),
                "train/recon_loss": epoch_losses[1].item(),
                "train/kl_loss": epoch_losses[2].item(),
                "train/spectral_loss": epoch_losses[3].item(),
                "train/bn_loss": epoch_losses[4].item()
            },
            step=global_step
        )

    return global_step


def epoch_validation(dataloader: DataLoader, model: AutoEncoder, global_step: int, run: wandb.run = None):

    # for logging: loss, rec, kl
    epoch_losses = torch.empty((0, 3), device=f"cuda:{LOCAL_RANK}")

    for step, x in enumerate(tqdm(dataloader)):

        x = x[0].to(torch.float32)

        logits, kl_all = model(x)
        reconstructions = DiscMixLogistic(logits, img_channels=3, num_bits=8).log_prob(x)

        # reconstruction loss
        rec_loss = - torch.sum(reconstructions, dim=1)

        final_kl, _, _ = kl_balancer(kl_all, balance=False)
        loss = rec_loss + final_kl

        # save all loss terms to rank 0
        losses = torch.stack([loss.mean(), rec_loss.mean(), final_kl.mean()], dim=0)

        # send to all devices and take mean
        dist.all_reduce(losses, op=dist.ReduceOp.SUM)
        losses = losses / WORLD_SIZE

        # get full Batch mean losses for iteration.
        epoch_losses, _ = pack([epoch_losses, losses], '* n')

    # log epoch loss
    epoch_losses = torch.mean(epoch_losses, dim=0)

    if WORLD_RANK == 0 and run is not None:

        run.log(
            {
                "validation/loss": epoch_losses[0].item(),
                "validation/recon_loss": epoch_losses[1].item(),
                "validation/kl_loss": epoch_losses[2].item(),
            },
            step=global_step
        )


def main(args: argparse.Namespace, config: dict):

    # init wandb logger
    log_to_wandb = bool(args.logging)
    project_name = str(args.wandb_project)
    wandb_id = args.wandb_id

    if WORLD_RANK == 0:
        run = wandb.init(project=project_name, name=args.run_name, mode="offline" if not log_to_wandb else "online",
                         resume="must" if wandb_id is not None else None, id=wandb_id)
    else:
        run = None

    train_conf = config['training']

    setup(train_conf)

    # Get data loaders.
    train_loader, val_loader, train_augmentations = prepare_data(LOCAL_RANK, WORLD_SIZE, args.data_path, config)

    # create model and move it to GPU with id rank
    model = AutoEncoder(config['autoencoder'], config['resolution'])

    # load checkpoint if resume
    if args.resume_from is not None:

        if WORLD_RANK == 0:
            print(f'[INFO] Loading checkpoint from: {args.resume_from}')

        checkpoint = torch.load(args.resume_from, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(LOCAL_RANK)
        global_step = checkpoint['global_step']
        init_epoch = checkpoint['epoch']
        optimizer_state = checkpoint['optimizer']
        grad_scaler_state = checkpoint['grad_scaler']

        if WORLD_RANK == 0:
            print(f'[INFO] Start from Epoch: {init_epoch} - Step: {global_step}')

    else:
        global_step, init_epoch = 0, 0
        optimizer_state = None
        grad_scaler_state = None

        model = model.to(LOCAL_RANK)

        if WORLD_RANK == 0:
            print(summary(model, torch.zeros((1,) + tuple(config['resolution']), device=f'cuda:{LOCAL_RANK}'),
                          show_input=False))

    # find final learning rate
    learning_rate = float(train_conf['base_lr'])
    min_learning_rate = float(train_conf['min_lr'])
    weight_decay = float(train_conf['weight_decay'])
    eps = float(train_conf['eps'])

    if WORLD_RANK == 0:
        print(f'[INFO] final learning rate: {learning_rate}')
        print(f'[INFO] final min learning rate: {min_learning_rate}')

    # ddp model, optimizer, scheduler, scaler
    ddp_model = DDP(model, device_ids=[LOCAL_RANK])

    optimizer = torch.optim.Adamax(ddp_model.parameters(), learning_rate, weight_decay=weight_decay, eps=eps)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    total_training_steps = len(train_loader) * train_conf['epochs']
    if train_conf['warmup_epochs'] is not None:
        warmup_steps = int(len(train_loader) * train_conf['warmup_epochs'])
        scheduler = LinearCosineScheduler(0, total_training_steps, learning_rate,
                                          min_learning_rate, warmup_steps)
    else:
        scheduler = CosineScheduler(0, total_training_steps, learning_rate, min_learning_rate)

    grad_scaler = GradScaler(2 ** 10)  # scale gradient for AMP
    if grad_scaler_state is not None:
        grad_scaler.load_state_dict(grad_scaler_state)

    for epoch in range(init_epoch, train_conf['epochs']):

        if WORLD_RANK == 0:
            print(f'[INFO] Epoch {epoch+1}/{train_conf["epochs"]}')
            run.log({'epoch': epoch}, step=global_step)

        # Training
        ddp_model.train()
        global_step = epoch_train(train_loader, train_augmentations, ddp_model.module, optimizer, scheduler,
                                  grad_scaler, train_conf["kl_anneal"], train_conf["spectral_regularization"],
                                  total_training_steps, global_step, run)

        # sync of all parameters
        # TODO after refactor of weight norm check if still needed!
        for p in ddp_model.module.parameters():
            dist.all_reduce(p.data, op=dist.ReduceOp.SUM)
            p.data /= WORLD_SIZE

        # Validation
        dist.barrier()
        eval_freq = 1 if train_conf["epochs"] <= 50 else 5

        if epoch % eval_freq == 0 or epoch == (train_conf["epochs"] - 1):

            if WORLD_RANK == 0:
                print('[INFO] Validating')

            ddp_model.eval()
            with torch.no_grad():

                if WORLD_RANK == 0:

                    num_samples = 8

                    # log reconstructions
                    x = next(iter(val_loader))[0][:num_samples].to(torch.float32)
                    b, c, h, w = x.shape

                    logits = ddp_model.module.autoencode(x, deterministic=True)
                    x_rec = DiscMixLogistic(logits, img_channels=3, num_bits=8).mean()

                    display, _ = pack([x, x_rec], '* c h w')
                    display = make_grid(display, nrow=b)
                    display = wandb.Image(display)
                    run.log({f"media/reconstructions": display}, step=global_step)

                    # log samples
                    for t in [0.7, 0.8, 0.9, 1.0]:
                        logits = ddp_model.module.sample(num_samples, t, device='cuda:0')
                        samples = DiscMixLogistic(logits, img_channels=3, num_bits=8).sample(t)
                        display = wandb.Image(make_grid(samples, nrow=num_samples))
                        run.log({f"media/samples tau={t:.2f}": display}, step=global_step)

                epoch_validation(val_loader, ddp_model.module, global_step, run)

            # Save checkpoint (after validation)
            if WORLD_RANK == 0:
                print(f'[INFO] Saving Checkpoint')

                checkpoint_dict = {
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'configuration': config,
                    'state_dict': ddp_model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'grad_scaler': grad_scaler.state_dict(),
                }
                ckpt_file = f"{args.checkpoint_base_path}/{args.run_name}/epoch={epoch:02d}.pt"
                torch.save(checkpoint_dict, ckpt_file)

            dist.barrier()

    if WORLD_RANK == 0:
        wandb.finish()

        print(f'[INFO] Saving Checkpoint')

        checkpoint_dict = {
            'epoch': train_conf["epochs"],
            'global_step': global_step,
            'configuration': config,
            'state_dict': ddp_model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'grad_scaler': grad_scaler.state_dict()
        }
        ckpt_file = f"{args.checkpoint_base_path}/{args.run_name}/last.pt"
        torch.save(checkpoint_dict, ckpt_file)

    cleanup()


if __name__ == '__main__':

    # on multinode cluster use mpi (enables to run only from master node)
    # on single_node local environment use torchrun
    # for debugging do not use anything (works only on 1 gpu)

    # torchrun --nproc_per_node=ngpus --nnodes=1 --node_rank=0 --master_addr='localhost'
    # --master_port=1234 main.py --> args

    # mpirun -np world_size -H ip_node_0:n_gpus,ip_node_1:n_gpus ... -x MASTER_ADDR=ip_master -x MASTER_PORT=1234
    # -bind-to none -map-by slot -mca pml ob1 -mca btl ^openib
    # python main.py --args

    # Environment variables set by torch.distributed.launch or mpirun
    if 'LOCAL_RANK' in os.environ:
        # launched with torch distributed run
        LOCAL_RANK = int(os.environ['LOCAL_RANK'])
        WORLD_SIZE = int(os.environ['WORLD_SIZE'])
        WORLD_RANK = int(os.environ['RANK'])
    elif 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        # launched with ompi
        LOCAL_RANK = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        WORLD_RANK = int(os.environ['OMPI_COMM_WORLD_RANK'])
    else:
        # launched with standard python, debugging only
        print('[INFO] DEBUGGING MODE on single gpu!')
        LOCAL_RANK = 0
        WORLD_SIZE = 1
        WORLD_RANK = 0

    if LOCAL_RANK == 0:
        print(f'[INFO] STARTING on NODE: {socket.gethostname()}')

    if WORLD_RANK == 0:
        print(f'[INFO] Total number of processes: {WORLD_SIZE}')

    cudnn.benchmark = True

    arguments = parse_args()

    try:
        main(arguments, get_model_conf(arguments.conf_file))
    except KeyboardInterrupt as e:
        wandb.finish()
        dist.destroy_process_group()
        raise e
