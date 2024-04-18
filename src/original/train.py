# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import torch
import numpy as np
import os

import torch.distributed as dist
import wandb
from einops import pack
from pytorch_model_summary import summary
from torch.multiprocessing import Process
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import make_grid

from model import AutoEncoder
from thirdparty.adamax import Adamax
import utils
import datasets

from fid.fid_score import compute_statistics_of_generator, load_statistics, calculate_frechet_distance
from fid.inception import InceptionV3


def main(args):

    # ensures that weight initializations are all the same
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logging = utils.Logger(args.global_rank, args.save)

    if args.global_rank == 0:
        run = wandb.init(project='nvae', name=f'original_{args.run_name}', mode="online")
    else:
        run = None

    # Get data loaders.
    train_queue, valid_queue, num_classes = datasets.get_loaders(args)
    args.num_total_iter = len(train_queue) * args.epochs
    warmup_iters = len(train_queue) * args.warmup_epochs
    # swa_start = len# load nvae pretrained cifar10(train_queue) * (args.epochs - 1)

    arch_instance = utils.get_arch_cells(args.arch_instance)

    model = AutoEncoder(args, arch_instance)
    model = model.cuda()

    if args.global_rank == 0:
        print(summary(model, torch.zeros((1, 3, 32, 32), device=f'cuda:{args.global_rank}'), show_input=False))

    logging.info('args = %s', args)
    logging.info('param size = %fM ', utils.count_parameters_in_M(model))
    logging.info('groups per scale: %s, total_groups: %d', model.groups_per_scale, sum(model.groups_per_scale))

    if args.fast_adamax:
        # Fast adamax has the same functionality as torch.optim.Adamax, except it is faster.
        cnn_optimizer = Adamax(model.parameters(), args.learning_rate,
                               weight_decay=args.weight_decay, eps=1e-3)
    else:
        cnn_optimizer = torch.optim.Adamax(model.parameters(), args.learning_rate,
                                           weight_decay=args.weight_decay, eps=1e-3)

    cnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        cnn_optimizer, float(args.epochs - args.warmup_epochs - 1), eta_min=args.learning_rate_min)
    grad_scalar = GradScaler(2**10)

    num_output = utils.num_output(args.dataset)
    bpd_coeff = 1. / np.log(2.) / num_output

    # if load
    checkpoint_file = os.path.join(args.save, 'checkpoint.pt')
    if args.cont_training:
        logging.info('loading the model.')
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        init_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        model = model.cuda()
        cnn_optimizer.load_state_dict(checkpoint['optimizer'])
        grad_scalar.load_state_dict(checkpoint['grad_scalar'])
        cnn_scheduler.load_state_dict(checkpoint['scheduler'])
        global_step = checkpoint['global_step']
    else:
        global_step, init_epoch = 0, 0

    for epoch in range(init_epoch, args.epochs):

        # update lrs.
        if args.distributed:
            train_queue.sampler.set_epoch(global_step + args.seed)
            valid_queue.sampler.set_epoch(0)

        if epoch > args.warmup_epochs:
            cnn_scheduler.step()

        # Logging.
        logging.info('epoch %d', epoch)
        if args.global_rank == 0:
            run.log({"epoch": epoch}, global_step)

        # Training.
        train_nelbo, global_step = train(train_queue, model, cnn_optimizer, grad_scalar, global_step, warmup_iters, logging, run)
        logging.info('train_nelbo %f', train_nelbo)

        model.eval()

        # generate samples less frequently
        eval_freq = 1 if args.epochs <= 50 else 5
        if epoch % eval_freq == 0 or epoch == (args.epochs - 1):

            with torch.no_grad():

                if args.global_rank == 0:

                    model.eval()
                    num_samples = 8

                    # log reconstructions
                    x = next(iter(valid_queue))[0][:num_samples].to(f"cuda:0")
                    b, c, h, w = x.shape

                    logits = model.decode(model.encode_deterministic(x))
                    x_rec = model.decoder_output(logits).mean()

                    display, _ = pack([x, x_rec], '* c h w')
                    display = make_grid(display, nrow=b)
                    display = wandb.Image(display)
                    run.log({f"media/reconstructions": display}, step=global_step)

                    # log samples
                    for t in [0.7, 0.8, 0.9, 1.0]:
                       logits = model.sample(num_samples, t)
                       samples = model.decoder_output(logits).sample()
                       display = wandb.Image(make_grid(samples, nrow=num_samples))
                       run.log({f"media/samples tau={t:.2f}": display}, step=global_step)

            valid_neg_log_p, valid_nelbo = test(valid_queue, model, step=global_step, args=args, logging=logging, run=run)
            logging.info('valid_nelbo %f', valid_nelbo)
            logging.info('valid neg log p %f', valid_neg_log_p)
            logging.info('valid bpd elbo %f', valid_nelbo * bpd_coeff)
            logging.info('valid bpd log p %f', valid_neg_log_p * bpd_coeff)

        save_freq = int(np.ceil(args.epochs / 100))
        if epoch % save_freq == 0 or epoch == (args.epochs - 1):
            if args.global_rank == 0:
                logging.info('saving the model.')
                torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                            'optimizer': cnn_optimizer.state_dict(), 'global_step': global_step,
                            'args': args, 'arch_instance': arch_instance, 'scheduler': cnn_scheduler.state_dict(),
                            'grad_scalar': grad_scalar.state_dict()}, checkpoint_file)

    # Final validation
    valid_neg_log_p, valid_nelbo = test(valid_queue, model, step=global_step, args=args, logging=logging, run=run)
    logging.info('final valid nelbo %f', valid_nelbo)
    logging.info('final valid neg log p %f', valid_neg_log_p)


def train(train_queue, model, cnn_optimizer, grad_scalar, global_step, warmup_iters, logging, run):

    alpha_i = utils.kl_balancer_coeff(num_scales=model.num_latent_scales,
                                      groups_per_scale=model.groups_per_scale, fun='square')
    nelbo = utils.AvgrageMeter()
    model.train()

    # for logging: loss, rec, kl, spectral, bn
    epoch_losses = torch.empty((0, 5), device=f"cuda:{args.global_rank}")

    for step, x in enumerate(train_queue):
        x = x[0] if len(x) > 1 else x
        x = x.cuda()

        # change bit length
        x = utils.pre_process(x, args.num_x_bits)

        # warm-up lr
        if global_step < warmup_iters:
            lr = args.learning_rate * float(global_step) / warmup_iters
            for param_group in cnn_optimizer.param_groups:
                param_group['lr'] = lr

        # sync parameters, it may not be necessary
        if step % 100 == 0:
            utils.average_params(model.parameters(), args.distributed)

        cnn_optimizer.zero_grad()
        with autocast():
            logits, log_q, log_p, kl_all, kl_diag = model(x)

            output = model.decoder_output(logits)
            kl_coeff = utils.kl_coeff(global_step, args.kl_anneal_portion * args.num_total_iter,
                                      args.kl_const_portion * args.num_total_iter, args.kl_const_coeff)

            recon_loss = utils.reconstruction_loss(output, x, crop=model.crop_output)
            balanced_kl, kl_coeffs, kl_vals = utils.kl_balancer(kl_all, kl_coeff, kl_balance=True, alpha_i=alpha_i)

            # TODO changed this adding non elbo weight!!
            nelbo_batch = recon_loss + balanced_kl * args.non_elbo_weight
            loss = torch.mean(nelbo_batch)
            norm_loss = model.compute_sr_loss() * args.non_elbo_weight
            bn_loss = model.batchnorm_loss() * args.non_elbo_weight
            # get spectral regularization coefficient (lambda)
            if args.weight_decay_norm_anneal:
                assert args.weight_decay_norm_init > 0 and args.weight_decay_norm > 0, 'init and final wdn should be positive.'
                wdn_coeff = (1. - kl_coeff) * np.log(args.weight_decay_norm_init) + kl_coeff * np.log(args.weight_decay_norm)
                wdn_coeff = np.exp(wdn_coeff)
            else:
                wdn_coeff = args.weight_decay_norm

            loss += norm_loss * wdn_coeff + bn_loss * wdn_coeff

        grad_scalar.scale(loss).backward()
        utils.average_gradients(model.parameters(), args.distributed)
        grad_scalar.step(cnn_optimizer)
        grad_scalar.update()
        nelbo.update(loss.data, 1)

        # to log at each step
        if args.global_rank == 0:

            log_dict = {'lr': cnn_optimizer.param_groups[0]['lr'], 'KL beta': kl_coeff, 'Lambda': wdn_coeff}
            for i, (k, v) in enumerate(zip(kl_coeffs.detach().cpu().numpy(), kl_vals.detach().cpu().numpy())):
                log_dict[f'KL gamma {i}'] = k
                log_dict[f'KL term {i}'] = v

            run.log(log_dict, step=global_step)

        # save all loss terms to rank 0
        losses = torch.stack(
            [loss, recon_loss.mean(), balanced_kl.mean(), norm_loss * wdn_coeff, bn_loss * wdn_coeff],
            dim=0).detach()

        if not args.global_rank == 0:
            dist.gather(tensor=losses, dst=0)
        else:
            batch_losses = [torch.zeros_like(losses) for _ in range(args.num_process_per_node)]
            dist.gather(gather_list=batch_losses, tensor=losses)

            # loss, rec, kl, spectral, bn
            epoch_losses, _ = pack([epoch_losses, torch.mean(torch.stack(batch_losses, dim=0), dim=0).unsqueeze(0)],
                                   '* n')

        if (global_step + 1) % 100 == 0:

            utils.average_tensor(nelbo.avg, args.distributed)
            logging.info('train %d %f', global_step, nelbo.avg)

            total_active = 0
            for i, kl_diag_i in enumerate(kl_diag):
                utils.average_tensor(kl_diag_i, args.distributed)
                num_active = torch.sum(kl_diag_i > 0.1).detach()
                total_active += num_active

        global_step += 1

    # log epoch loss
    if args.global_rank == 0:
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

    utils.average_tensor(nelbo.avg, args.distributed)
    return nelbo.avg, global_step


def test(valid_queue, model, step, args, logging, run):

    if args.distributed:
        dist.barrier()

    # for logging: loss, rec, kl
    epoch_losses = torch.empty((0, 3), device=f"cuda:{args.global_rank}")

    nelbo_avg = utils.AvgrageMeter()
    neg_log_p_avg = utils.AvgrageMeter()

    model.eval()

    for _, x in enumerate(valid_queue):

        x = x[0] if len(x) > 1 else x
        x = x.cuda()

        # change bit length
        x = utils.pre_process(x, args.num_x_bits)

        with torch.no_grad():
            nelbo, log_iw = [], []

            logits, log_q, log_p, kl_all, _ = model(x)
            output = model.decoder_output(logits)
            recon_loss = utils.reconstruction_loss(output, x, crop=model.crop_output)
            balanced_kl, _, _ = utils.kl_balancer(kl_all, kl_balance=False)
            nelbo_batch = recon_loss + balanced_kl
            nelbo.append(nelbo_batch)
            log_iw.append(utils.log_iw(output, x, log_q, log_p, crop=model.crop_output))

            # save all loss terms to rank 0
            losses = torch.stack([nelbo_batch.mean(), recon_loss.mean(), balanced_kl.mean()], dim=0)

            if not args.global_rank == 0:
                dist.gather(tensor=losses, dst=0)
            else:
                batch_losses = [torch.zeros_like(losses) for _ in range(args.num_process_per_node)]
                dist.gather(gather_list=batch_losses, tensor=losses)

                # get full Batch mean losses for iteration.
                epoch_losses, _ = pack(
                    [epoch_losses, torch.mean(torch.stack(batch_losses, dim=0), dim=0).unsqueeze(0)],
                    '* n')


            nelbo = torch.mean(torch.stack(nelbo, dim=1))
            log_p = torch.mean(torch.logsumexp(torch.stack(log_iw, dim=1), dim=1) - np.log(1))

        nelbo_avg.update(nelbo.data, x.size(0))
        neg_log_p_avg.update(- log_p.data, x.size(0))

    utils.average_tensor(nelbo_avg.avg, args.distributed)
    utils.average_tensor(neg_log_p_avg.avg, args.distributed)
    if args.distributed:
        # block to sync
        dist.barrier()
    logging.info('val, step: %d, NELBO: %f, neg Log p %f', step, nelbo_avg.avg, neg_log_p_avg.avg)

    # mine
    if args.global_rank == 0:
        epoch_losses = epoch_losses.mean(dim=0)
        run.log(
            {
                "validation/loss": epoch_losses[0].item(),
                "validation/recon_loss": epoch_losses[1].item(),
                "validation/kl_loss": epoch_losses[2].item(),
            },
            step=step
        )

    return neg_log_p_avg.avg, nelbo_avg.avg


def create_generator_vae(model, batch_size, num_total_samples):
    num_iters = int(np.ceil(num_total_samples / batch_size))
    for i in range(num_iters):
        with torch.no_grad():
            logits = model.sample(batch_size, 1.0)
            output = model.decoder_output(logits)
            output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) else output.mean()
        yield output_img.float()


def test_vae_fid(model, args, total_fid_samples):
    dims = 2048
    device = 'cuda'
    num_gpus = args.num_process_per_node * args.num_proc_node
    num_sample_per_gpu = int(np.ceil(total_fid_samples / num_gpus))

    g = create_generator_vae(model, args.batch_size, num_sample_per_gpu)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], model_dir=args.fid_dir).to(device)
    m, s = compute_statistics_of_generator(g, model, args.batch_size, dims, device, max_samples=num_sample_per_gpu)

    # share m and s
    m = torch.from_numpy(m).cuda()
    s = torch.from_numpy(s).cuda()
    # take average across gpus
    utils.average_tensor(m, args.distributed)
    utils.average_tensor(s, args.distributed)

    # convert m, s
    m = m.cpu().numpy()
    s = s.cpu().numpy()

    # load precomputed m, s
    path = os.path.join(args.fid_dir, args.dataset + '.npz')
    m0, s0 = load_statistics(path)

    fid = calculate_frechet_distance(m0, s0, m, s)
    return fid


def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '6021'
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(args)
    cleanup()


def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('encoder decoder examiner')

    parser.add_argument('--non_elbo_weight', type=float, default=1.)
    parser.add_argument('--run_name', type=str)

    # experimental results
    parser.add_argument('--root', type=str, default='/tmp/nasvae/expr',
                        help='location of the results')
    parser.add_argument('--save', type=str, default='exp',
                        help='id used for storing intermediate results')
    # data
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['cifar10', 'mnist', 'omniglot', 'celeba_64', 'celeba_256',
                                 'imagenet_32', 'ffhq', 'lsun_bedroom_128', 'stacked_mnist',
                                 'lsun_church_128', 'lsun_church_64'],
                        help='which dataset to use')
    parser.add_argument('--data', type=str, default='/tmp/nasvae/data',
                        help='location of the data corpus')
    # optimization
    parser.add_argument('--batch_size', type=int, default=200,
                        help='batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                        help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=1e-4,
                        help='min learning rate')
    parser.add_argument('--weight_decay', type=float, default=3e-4,
                        help='weight decay')
    parser.add_argument('--weight_decay_norm', type=float, default=0.,
                        help='The lambda parameter for spectral regularization.')
    parser.add_argument('--weight_decay_norm_init', type=float, default=10.,
                        help='The initial lambda parameter')
    parser.add_argument('--weight_decay_norm_anneal', action='store_true', default=False,
                        help='This flag enables annealing the lambda coefficient from '
                             '--weight_decay_norm_init to --weight_decay_norm.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='num of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='num of training epochs in which lr is warmed up')
    parser.add_argument('--fast_adamax', action='store_true', default=False,
                        help='This flag enables using our optimized adamax.')
    parser.add_argument('--arch_instance', type=str, default='res_mbconv',
                        help='path to the architecture instance')
    # KL annealing
    parser.add_argument('--kl_anneal_portion', type=float, default=0.3,
                        help='The portions epochs that KL is annealed')
    parser.add_argument('--kl_const_portion', type=float, default=0.0001,
                        help='The portions epochs that KL is constant at kl_const_coeff')
    parser.add_argument('--kl_const_coeff', type=float, default=0.0001,
                        help='The constant value used for min KL coeff')
    # Flow params
    parser.add_argument('--num_nf', type=int, default=0,
                        help='The number of normalizing flow cells per groups. Set this to zero to disable flows.')
    parser.add_argument('--num_x_bits', type=int, default=8,
                        help='The number of bits used for representing data for colored images.')
    # latent variables
    parser.add_argument('--num_latent_scales', type=int, default=1,
                        help='the number of latent scales')
    parser.add_argument('--num_groups_per_scale', type=int, default=10,
                        help='number of groups of latent variables per scale')
    parser.add_argument('--num_latent_per_group', type=int, default=20,
                        help='number of channels in latent variables per group')
    parser.add_argument('--ada_groups', action='store_true', default=False,
                        help='Settings this to true will set different number of groups per scale.')
    parser.add_argument('--min_groups_per_scale', type=int, default=1,
                        help='the minimum number of groups per scale.')
    # encoder parameters
    parser.add_argument('--num_channels_enc', type=int, default=32,
                        help='number of channels in encoder')
    parser.add_argument('--num_preprocess_blocks', type=int, default=2,
                        help='number of preprocessing blocks')
    parser.add_argument('--num_preprocess_cells', type=int, default=3,
                        help='number of cells per block')
    parser.add_argument('--num_cell_per_cond_enc', type=int, default=1,
                        help='number of cell for each conditional in encoder')
    # decoder parameters
    parser.add_argument('--num_channels_dec', type=int, default=32,
                        help='number of channels in decoder')
    parser.add_argument('--num_postprocess_blocks', type=int, default=2,
                        help='number of postprocessing blocks')
    parser.add_argument('--num_postprocess_cells', type=int, default=3,
                        help='number of cells per block')
    parser.add_argument('--num_cell_per_cond_dec', type=int, default=1,
                        help='number of cell for each conditional in decoder')
    parser.add_argument('--num_mixture_dec', type=int, default=10,
                        help='number of mixture components in decoder. set to 1 for Normal decoder.')
    # NAS
    parser.add_argument('--use_se', action='store_true', default=False,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--res_dist', action='store_true', default=False,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--cont_training', action='store_true', default=False,
                        help='This flag enables training from an existing checkpoint.')
    # DDP.
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--global_rank', type=int, default=0,
                        help='rank of process among all the processes')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed used for initialization')
    args = parser.parse_args()
    args.save = args.root + '/eval-' + args.save
    utils.create_exp_dir(args.save)

    size = args.num_process_per_node

    try:
        if size > 1:
            args.distributed = True
            processes = []
            for rank in range(size):
                args.local_rank = rank
                global_rank = rank + args.node_rank * args.num_process_per_node
                global_size = args.num_proc_node * args.num_process_per_node
                args.global_rank = global_rank
                print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
                p = Process(target=init_processes, args=(global_rank, global_size, main, args))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
        else:
            # for debugging
            print('starting in debug mode')
            args.distributed = True
            init_processes(0, size, main, args)
    except KeyboardInterrupt:
        wandb.finish()
        cleanup()

