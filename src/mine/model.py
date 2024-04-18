import math
from collections import OrderedDict

from einops import pack, rearrange, einsum

import torch
from torch import nn

from src.hl_autoencoders.NVAE.mine.modules import ResidualCellEncoder, EncCombinerCell, NFBlock, ResidualCellDecoder, DecCombinerCell, \
    ARConv2d, Conv2D
from src.hl_autoencoders.NVAE.mine.distributions import Normal


class AutoEncoder(nn.Module):
    def __init__(self, ae_args: dict, resolution: tuple, use_SE: bool = True):
        """
        :param ae_args: hierarchical args
        :param use_SE: use Squeeze and Excitation
        """

        super().__init__()

        # ######################################################################################
        # general params
        self.img_channels, self.image_resolution, _ = resolution
        self.base_channels, self.ch_multiplier = ae_args['initial_channels'], 1
        self.use_SE = use_SE

        # pre and post processing
        self.n_preprocessing_blocks = ae_args['num_pre-post_process_blocks']
        self.n_cells_per_preprocessing_block = ae_args['num_pre-post_process_cells']
        self.n_postprocessing_blocks = ae_args['num_pre-post_process_blocks']
        self.n_cells_per_postprocessing_block = ae_args['num_pre-post_process_cells']

        # logistic mixture
        self.num_mixtures = ae_args['num_logistic_mixtures']

        # encoder - decoder
        self.num_scales = ae_args['num_scales']
        self.groups_per_scale = [
            max(ae_args['min_groups_per_scale'], ae_args['num_groups_per_scale'] // (2 ** i))
            if ae_args['is_adaptive'] else
            ae_args['num_groups_per_scale']
            for i in range(self.num_scales)
        ]  # different on each scale if is_adaptive (scale N has more groups, scale 0 has less)
        self.groups_per_scale.reverse()

        self.num_cells_per_group = ae_args['num_cells_per_group']

        # latent variables and sampling
        self.num_latent_per_group = ae_args['num_latent_per_group']
        self.num_nf_cells = ae_args['num_nf_cells']
        self.use_nf = self.num_nf_cells is not None

        # loss coefficients
        kl_alpha = [(2 ** i) ** 2 / self.groups_per_scale[self.num_scales - i - 1] *
                    torch.ones(self.groups_per_scale[self.num_scales - i - 1])
                    for i in range(self.num_scales)]
        kl_alpha = torch.cat(kl_alpha, dim=0)
        self.kl_alpha = (kl_alpha / torch.min(kl_alpha)).unsqueeze(0)  # normalize to min 1

        # ######################################################################################
        # initial learned constant
        self.scaling_factor = 2 ** (self.n_preprocessing_blocks + self.num_scales - 1)
        c, r = int(self.scaling_factor * self.base_channels), self.image_resolution // self.scaling_factor
        self.const_prior = nn.Parameter(torch.rand(size=(1, c, r, r)), requires_grad=True)

        # structure
        self.preprocessing_block = self._init_preprocessing()

        self.encoder_tower, self.encoder_combiners, self.encoder_0 = self._init_encoder()

        self.enc_sampler, self.dec_sampler, self.nf_cells = self._init_samplers()

        self.decoder_tower, self.decoder_combiners = self._init_decoder()

        self.postprocessing_block = self._init_postprocessing()

        self.to_logits = self._init_to_logits()

        # ######################################################################################

        # compute values for spectral and batch norm regularization
        self._count_conv_and_batch_norm_layers()

        # left/right singular vectors used for estimating the largest singular values of each weight.
        self.sr_u = {}
        self.sr_v = {}
        self.num_power_iter = 4  # number of iterations for estimating each u and v.

    def _init_preprocessing(self):
        """
        Returns the preprocessing module, structured as:
                init_conv that changes num channels form self.img_channels to self.base_channels
                N blocks of M cells each.
                Each block doubles the number of channels and downsamples H, W by a factor of 2
        """

        preprocessing = OrderedDict()
        preprocessing['init_conv'] = Conv2D(self.img_channels, self.base_channels, kernel_size=3, padding=1,
                                            bias=True, weight_norm=True)

        for b in range(self.n_preprocessing_blocks):

            block = OrderedDict()

            for c in range(self.n_cells_per_preprocessing_block):

                is_last = (c == self.n_cells_per_preprocessing_block - 1)

                if not is_last:
                    # standard cell
                    channels = self.base_channels * self.ch_multiplier
                    cell = ResidualCellEncoder(channels, channels, downsampling=False, use_SE=self.use_SE)
                else:
                    # cell with downsampling (double channels)
                    channels = self.base_channels * self.ch_multiplier
                    cell = ResidualCellEncoder(channels, channels * 2, downsampling=True, use_SE=self.use_SE)
                    self.ch_multiplier *= 2

                block[f'cell_{c}'] = cell

            preprocessing[f'block_{b}'] = nn.Sequential(block)
        return nn.Sequential(preprocessing)

    def _init_encoder(self):
        """

        Returns the encoder tower module, the encoder combiner and the encoder 0 module.

        encoder tower has N Scales, indexed from N-1 to 0 (in the sense that N-1 scale is the closest to the input).
        each scale has M groups, where M can optionally change in each scale if the is_adaptive parameter is True.
        each group has K residuals cells, structured as Fig 2b. of the paper

        For each scale > 0, a downsampling operator applies and number of channels is doubled
        For each group (except last one of scale 0) one latent variable will be produced, through the encoder_combiners
        module

        """

        encoder_tower = nn.ModuleList()
        encoder_combiners = nn.ModuleList()

        for s in range(self.num_scales - 1, -1, -1):

            scale = nn.ModuleList()
            channels = int(self.base_channels * self.ch_multiplier)

            # add groups, cells to scale and combiner at the end
            for g in range(self.groups_per_scale[s] - 1, -1, -1):

                group = nn.ModuleList()

                # add cells for this group
                for c in range(self.num_cells_per_group):
                    cell = ResidualCellEncoder(channels, channels, downsampling=False, use_SE=self.use_SE)
                    group.add_module(f'cell_{c}', cell)

                # at the end of each group, except last group and scale, add combiner (for sampling z)
                if not (s == 0 and g == 0):
                    cell = EncCombinerCell(in_channels=channels, out_channels=channels)
                    encoder_combiners.add_module(f'combiner_{s}:{g}', cell)

                # add group to scale
                scale.add_module(f'group_{g}', group)

            # final downsampling (except for last scale)
            if s > 0:
                # cell with downsampling (double channels)
                cell = ResidualCellEncoder(channels, channels * 2, downsampling=True, use_SE=self.use_SE)
                scale.add_module(f'downsampling', cell)
                self.ch_multiplier *= 2

            encoder_tower.add_module(f'scale_{s}', scale)

        # encoder 0 takes the final output of encoder tower
        channels = int(self.base_channels * self.ch_multiplier)
        encoder_0 = nn.Sequential(
            nn.ELU(),
            Conv2D(channels, channels, kernel_size=1, bias=True, weight_norm=True),
            nn.ELU())

        return encoder_tower, encoder_combiners, encoder_0

    def _init_samplers(self):
        """
        Returns encoder_sampler, decoder_sampler and optional NF cells for encoders
        encoder sampler (1 per latent variable) takes encoder_combiner output and gives mu, sigma params.
        decoder sampler same but for decoder
        NF cells, if specified, will be used after z_i sampling, only when auto-encoding
        """

        enc_sampler, dec_sampler = nn.ModuleList(), nn.ModuleList()
        nf_cells = nn.ModuleList() if self.use_nf else None

        ch_multiplier = self.ch_multiplier

        for s in range(self.num_scales):

            channels = int(self.base_channels * ch_multiplier)
            for g in range(self.groups_per_scale[s]):

                # encoder sampler
                enc_sampler.add_module(f'sampler_{s}:{g}',
                                       Conv2D(channels, 2 * self.num_latent_per_group,
                                              kernel_size=3, padding=1, bias=True, weight_norm=True))

                # build [optional] NF
                if self.use_nf:
                    this_nf_blocks = []
                    for n in range(self.num_nf_cells):
                        this_nf_blocks.append(NFBlock(self.num_latent_per_group))
                    nf_cells.add_module(f'nf_{s}:{g}', torch.nn.Sequential(*this_nf_blocks))

                # decoder samplers (first group uses standard gaussian)
                if not (s == 0 and g == 0):
                    dec_sampler.add_module(f'sampler_{s}:{g}',
                                           nn.Sequential(
                                               nn.ELU(),
                                               Conv2D(channels, 2 * self.num_latent_per_group,
                                                      kernel_size=1, padding=0, bias=True, weight_norm=True)
                                           )
                                           )

            ch_multiplier /= 2

        return enc_sampler, dec_sampler, nf_cells

    def _init_decoder(self):

        decoder_tower = nn.ModuleList()
        decoder_combiners = nn.ModuleList()

        for s in range(self.num_scales):

            scale = nn.ModuleList()
            channels = int(self.base_channels * self.ch_multiplier)

            for g in range(self.groups_per_scale[s]):

                group = nn.ModuleList()

                if not (s == 0 and g == 0):

                    # add cells for this group
                    for c in range(self.num_cells_per_group):
                        cell = ResidualCellDecoder(channels, channels, upsampling=False, use_SE=self.use_SE)
                        group.add_module(f'cell_{c}', cell)

                    scale.add_module(f'group_{g}', group)

                cell = DecCombinerCell(channels, self.num_latent_per_group, channels)
                decoder_combiners.add_module(f'combiner_{s}:{g}', cell)

            # upsampling cell at scale end (except for last)
            if s < self.num_scales - 1:
                cell = ResidualCellDecoder(channels, channels // 2, upsampling=True, use_SE=self.use_SE)
                scale.add_module(f'upsampling', cell)

                self.ch_multiplier /= 2

            decoder_tower.add_module(f'scale_{s}', scale)

        return decoder_tower, decoder_combiners

    def _init_postprocessing(self):

        postprocessing = OrderedDict()

        for b in range(self.n_postprocessing_blocks):

            block = OrderedDict()

            for c in range(self.n_cells_per_postprocessing_block):

                is_first = (c == 0)

                channels = int(self.base_channels * self.ch_multiplier)
                if not is_first:
                    # standard cell
                    cell = ResidualCellDecoder(channels, channels, upsampling=False, use_SE=self.use_SE, hidden_mul=3)
                else:
                    # cell with downsampling (double channels)
                    cell = ResidualCellDecoder(channels, channels // 2, upsampling=True, use_SE=self.use_SE,
                                               hidden_mul=3)
                    self.ch_multiplier /= 2

                block[f'cell_{c}'] = cell

            postprocessing[f'block_{b}'] = nn.Sequential(block)

        return nn.Sequential(postprocessing)

    def _init_to_logits(self):

        in_channels = int(self.base_channels * self.ch_multiplier)

        # OUT --> B, N_MIX + N_MIX * 3, C = 3, H, W
        # first n_mix are logits, then for each channel logistic_idx, means and sigmas
        out_channels = int(self.num_mixtures + self.num_mixtures * 3 * self.img_channels)

        to_logits = nn.Sequential(
            nn.ELU(),
            Conv2D(in_channels, out_channels, kernel_size=3, padding=1, bias=True, weight_norm=True)
        )

        return to_logits

    def _count_conv_and_batch_norm_layers(self):
        """
        Save all BN and Conv2D layers for quick access during BN and Spectral Regularization
        """

        self.all_conv_layers = []
        self.all_bn_layers = []

        for n, layer in self.named_modules():

            if isinstance(layer, Conv2D) or isinstance(layer, ARConv2d):
                self.all_conv_layers.append(layer)

            if isinstance(layer, nn.SyncBatchNorm):
                self.all_bn_layers.append(layer)

    def compute_sr_loss(self):
        """
        From Section 3.2 of NVAE paper:
            "To bound KL, we need to ensure that the encoder output does not change dramatically as its input changes.
            This notion of smoothness is characterized by the Lipschitz constant. We hypothesise that by regularizing
            the Lipschitz constant we can ensure that the latent codes predicted by the encoder remain bounded,
            resulting in stable KL minimization"

        What is the Lipschitz constant ?
            https://en.wikipedia.org/wiki/Lipschitz_continuity
            Intuitively, by regularizing the Lipschitz constant we ensure that the Encoder is continuous,
            avoiding large KL terms in the loss.

        Note -> in the code, SR is applied to both Encoder and Decoder, as the authors hypothesise that:
                "a sharp Decoder may require a sharp Encoder, causing more instability in training."
                Appendix A, annealing lambda.

        How to regularize the Lipschitz constant ?
            The idea comes from this paper: https://arxiv.org/pdf/1802.05957.pdf
            In Section 2.1, given Wl the weight matrix of Layer l and al the activation function of layer l, if
            the Lipschitz Norm of al is 1, then the Lipschitz norm ||f||Lp of the function f (e.g. the Encoder), can be
            bounded as:
                ||f||Lp <= PROD_{l=1}^L \sigma(Wl)   (Equation 7 of the paper).

            About the Lipschitz Norm:
            https://math.stackexchange.com/questions/2692579/is-lipschitz-norm-the-other-name-for-lipschitz-constant

            What is \sigma(Wl) ? It is the Spectral Norm of matrix Wl, equal to its largest singular value.
            https://math.stackexchange.com/questions/188202/meaning-of-the-spectral-norm-of-a-matrix#:~:text=The%20spectral%20norm%20(also%20know,can%20'stretch'%20a%20vector.

            Going back to Section 3.2 of NVAE paper:
            "Formally, we add the term Lsr = \lambda SUM_i si, where si is the largest singular value of the ith
            convolutional layer".

        How to estimate the singular values ?
            Since computing the singular values of a matrix can be computationally demanding, they use
            "power iteration method" (check section 3.2 and Appendix A of https://arxiv.org/pdf/1802.05957.pdf).

            The idea:
                1. \sigma(W) can be approximated by u^T W v,
                    where u and v are the first left and right singular vectors.
                2. u and v can be estimated with the iterative procedure:
                    for i in num_iterations:
                        u = norm_2( W v)
                        v = norm_2(u^T W)
                3. the initialization for u and v is uniform.

        """

        # get all convolutional weights by shape.
        weights = {}
        for w in self.all_conv_layers:
            # TODO change this if we change normalization of convs!
            weight = rearrange(w.weight_normalized, 'c x y z -> c (x y z)')
            if weight.shape not in weights:
                weights[weight.shape] = []

            weights[weight.shape].append(weight)

        # compute sr_loss for each weight
        sr_loss = 0
        for k, w in weights.items():

            # weights having the same shape can be processed together to fasten computation.
            w = torch.stack(w, dim=0)

            with torch.no_grad():

                # how many iterative updates to u, v
                num_iter = self.num_power_iter

                # first time (initialization of u and v according to uniform distribution)
                if k not in self.sr_u:

                    # increase the number of iterations for the first time only
                    num_iter = 10 * self.num_power_iter

                    # initialize u with same rows as w
                    self.sr_u[k] = nn.functional.normalize(
                        torch.ones_like(w)[:, :, 0].normal_(0, 1),
                        dim=1, eps=1e-3)

                    # initialize v with same rows as w
                    self.sr_v[k] = nn.functional.normalize(
                        torch.ones_like(w)[:, 0, :].normal_(0, 1),
                        dim=1, eps=1e-3)

                # update u, v iteratively
                for j in range(num_iter):

                    # v = norm_2(u^T W)
                    self.sr_v[k] = nn.functional.normalize(
                        einsum(self.sr_u[k], w, 'n r, n r c -> n c'),
                        dim=1, eps=1e-3)

                    # u = norm_2( W v)
                    self.sr_u[k] = nn.functional.normalize(
                        einsum(w, self.sr_v[k], 'n r c, n c -> n r'),
                        dim=1, eps=1e-3)

            # sigma of all weights with this shape = uT w v
            sigma = einsum(self.sr_u[k], w, self.sr_v[k], 'n r, n r c, n c -> n')
            sr_loss += torch.sum(sigma)  # sum on n

        return sr_loss

    def batch_norm_loss(self):
        """
        batch norm regularization term (also multiplied by λ)
        """
        loss = 0
        for l in self.all_bn_layers:
            if l.affine:
                loss += torch.max(torch.abs(l.weight))

        return loss

    def forward(self, gt_images: torch.Tensor):
        """
        :param gt_images: images in 0__1 range
        :return:
        """

        b = gt_images.shape[0]
        kl_losses = torch.empty((b, 0), device=gt_images.device)

        # preprocessing phase
        normalized_images = (gt_images * 2) - 1.0  # in range -1 1
        x = self.preprocessing_block(normalized_images)

        # encoding tower phase
        encoder_combiners_x = {}

        for s in range(self.num_scales - 1, -1, -1):

            scale = self.encoder_tower.get_submodule(f'scale_{s}')

            for g in range(self.groups_per_scale[s]):

                group = scale.get_submodule(f'group_{g}')

                for c in range(self.num_cells_per_group):
                    x = group.get_submodule(f'cell_{c}')(x)

                # add intermediate x (will be used as combiner input for this scale)
                if not (s == 0 and g == 0):
                    encoder_combiners_x[f'{s}:{g}'] = x

            if s > 0:
                x = scale.get_submodule(f'downsampling')(x)

        # encoder 0
        x = self.encoder_0(x)

        # obtain q(z_0|x), p(z_0) for KL loss, sample z_0

        # encoder q(z_0|x)
        mu_q, log_sig_q = torch.chunk(self.enc_sampler.get_submodule('sampler_0:0')(x), 2, dim=1)
        dist_enc = Normal(mu_q, log_sig_q)
        z_0, _ = dist_enc.sample()  # uses reparametrization trick

        # prior p(z_0) is a standard Gaussian (log_sigma 0 --> sigma = 1.)
        dist_dec = Normal(torch.zeros_like(mu_q), torch.zeros_like(log_sig_q))

        # apply normalizing flows
        if self.use_nf:
            log_enc = dist_enc.log_p(z_0)
            z_0 = self.nf_cells.get_submodule('nf_0:0')(z_0)
            log_dec = dist_dec.log_p(z_0)
            kl_0 = log_enc - log_dec  # with NF can't use closed form for kl
        else:
            kl_0 = dist_enc.kl(dist_dec)  # kl(q, p)

        kl_losses, _ = pack([kl_losses, torch.sum(kl_0, dim=[1, 2, 3]).unsqueeze(1)], 'b *')

        # decoding phase

        # start from constant prior
        x = self.const_prior.expand(b, -1, -1, -1)

        # first combiner (inject z_0)
        x = self.decoder_combiners.get_submodule('combiner_0:0')(x, z_0)

        for s in range(self.num_scales):

            scale = self.decoder_tower.get_submodule(f'scale_{s}')

            for g in range(self.groups_per_scale[s]):

                if not (s == 0 and g == 0):

                    # group forward
                    group = scale.get_submodule(f'group_{g}')

                    for c in range(self.num_cells_per_group):
                        x = group.get_submodule(f'cell_{c}')(x)

                    # obtain q(z_i|x, z_l>i), p(z_i|z_l>i) for KL loss, sample z_i
                    # extract params for p (conditioned on previous decoding)
                    mu_p, log_sig_p = torch.chunk(self.dec_sampler.get_submodule(f'sampler_{s}:{g}')(x), 2, dim=1)

                    # extract params for q (conditioned on encoder features and previous decoding)
                    enc_combiner = self.encoder_combiners.get_submodule(f'combiner_{s}:{g}')
                    enc_sampler = self.enc_sampler.get_submodule(f'sampler_{s}:{g}')
                    mu_q, log_sig_q = torch.chunk(
                        enc_sampler(enc_combiner(encoder_combiners_x[f'{s}:{g}'], x)),
                        2, dim=1)

                    # sample z_i as combination of encoder and decoder params
                    dist_enc = Normal(mu_p + mu_q, log_sig_p + log_sig_q)
                    z_i, _ = dist_enc.sample()

                    # log_p(z_i|z_j<i)
                    dist_dec = Normal(mu_p, log_sig_p)

                    # apply NF
                    if self.use_nf:
                        log_enc = dist_enc.log_p(z_i)
                        z_i = self.nf_cells.get_submodule(f'nf_{s}:{g}')(z_i)
                        log_dec = dist_dec.log_p(z_i)
                        kl_i = log_enc - log_dec
                    else:
                        kl_i = dist_enc.kl(dist_dec)

                    kl_losses, _ = pack([kl_losses, torch.sum(kl_i, dim=[1, 2, 3]).unsqueeze(1)], 'b *')

                    # combine x and z_i
                    x = self.decoder_combiners.get_submodule(f'combiner_{s}:{g}')(x, z_i)

            # upsampling at the end of each scale
            if s < self.num_scales - 1:
                x = scale.get_submodule('upsampling')(x)

        # postprocessing phase
        x = self.postprocessing_block(x)

        # get logits for mixture
        logits = self.to_logits(x)

        return logits, kl_losses

    def sample(self, num_samples: int, temperature: float, device: str = 'cpu'):

        # sample z_0
        samples_shape = (num_samples, self.num_latent_per_group,
                         self.image_resolution // self.scaling_factor, self.image_resolution // self.scaling_factor)
        dist = Normal(mu=torch.zeros(samples_shape, device=device), log_sigma=torch.zeros(samples_shape, device=device),
                      temp=temperature)
        z_0, _ = dist.sample()

        # get initial feature
        x = self.const_prior.expand(num_samples, -1, -1, -1)

        # first combiner (inject z_0)
        x = self.decoder_combiners.get_submodule('combiner_0:0')(x, z_0)

        for s in range(self.num_scales):

            scale = self.decoder_tower.get_submodule(f'scale_{s}')

            for g in range(self.groups_per_scale[s]):

                if not (s == 0 and g == 0):

                    # group forward
                    group = scale.get_submodule(f'group_{g}')

                    for c in range(self.num_cells_per_group):
                        x = group.get_submodule(f'cell_{c}')(x)

                    # extract params for p (conditioned on previous decoding)
                    mu_p, log_sig_p = torch.chunk(self.dec_sampler.get_submodule(f'sampler_{s}:{g}')(x), 2, dim=1)

                    # sample z_i
                    dist = Normal(mu_p, log_sig_p, temp=temperature)
                    z_i, _ = dist.sample()

                    # combine x and z_i
                    x = self.decoder_combiners.get_submodule(f'combiner_{s}:{g}')(x, z_i)

            # upsampling at the end of each scale
            if s < self.num_scales - 1:
                x = scale.get_submodule('upsampling')(x)

        # postprocessing phase
        x = self.postprocessing_block(x)

        # get logits for mixture
        logits = self.to_logits(x)

        return logits

    def encode(self, gt_images: torch.Tensor, deterministic: bool = False):
        """
        :param gt_images: images in 0__1 range
        :return:
        """

        chunks = []

        b = gt_images.shape[0]

        # preprocessing phase
        normalized_images = (gt_images * 2) - 1.0  # in range -1 1
        x = self.preprocessing_block(normalized_images)

        # encoding tower phase
        encoder_combiners_x = {}

        for s in range(self.num_scales - 1, -1, -1):

            scale = self.encoder_tower.get_submodule(f'scale_{s}')

            for g in range(self.groups_per_scale[s]):

                group = scale.get_submodule(f'group_{g}')

                for c in range(self.num_cells_per_group):
                    x = group.get_submodule(f'cell_{c}')(x)

                # add intermediate x (will be used as combiner input for this scale)
                if not (s == 0 and g == 0):
                    encoder_combiners_x[f'{s}:{g}'] = x

            if s > 0:
                x = scale.get_submodule(f'downsampling')(x)

        # encoder 0
        x = self.encoder_0(x)

        # encoder q(z_0|x)
        mu_q, log_sig_q = torch.chunk(self.enc_sampler.get_submodule('sampler_0:0')(x), 2, dim=1)
        dist = Normal(mu_q, log_sig_q)
        z_0 = dist.mu if deterministic else dist.sample()[0]

        chunks.append(z_0)

        # apply normalizing flows
        if self.use_nf:
            z_0 = self.nf_cells.get_submodule('nf_0:0')(z_0)

        # decoding phase

        # start from constant prior
        x = self.const_prior.expand(b, -1, -1, -1)

        # first combiner (inject z_0)
        x = self.decoder_combiners.get_submodule('combiner_0:0')(x, z_0)

        for s in range(self.num_scales):

            scale = self.decoder_tower.get_submodule(f'scale_{s}')

            for g in range(self.groups_per_scale[s]):

                if not (s == 0 and g == 0):

                    # group forward
                    group = scale.get_submodule(f'group_{g}')

                    for c in range(self.num_cells_per_group):
                        x = group.get_submodule(f'cell_{c}')(x)

                    # obtain q(z_i|x, z_l>i), p(z_i|z_l>i) for KL loss, sample z_i
                    # extract params for p (conditioned on previous decoding)
                    mu_p, log_sig_p = torch.chunk(self.dec_sampler.get_submodule(f'sampler_{s}:{g}')(x), 2, dim=1)

                    # extract params for q (conditioned on encoder features and previous decoding)
                    enc_combiner = self.encoder_combiners.get_submodule(f'combiner_{s}:{g}')
                    enc_sampler = self.enc_sampler.get_submodule(f'sampler_{s}:{g}')
                    mu_q, log_sig_q = torch.chunk(
                        enc_sampler(enc_combiner(encoder_combiners_x[f'{s}:{g}'], x)),
                        2, dim=1)

                    # sample z_i as combination of encoder and decoder params
                    dist = Normal(mu_p + mu_q, log_sig_p + log_sig_q)
                    z_i = dist.mu if deterministic else dist.sample()[0]

                    chunks.append(z_i)

                    # apply NF
                    if self.use_nf:
                        z_i = self.nf_cells.get_submodule(f'nf_{s}:{g}')(z_i)

                    # combine x and z_i
                    x = self.decoder_combiners.get_submodule(f'combiner_{s}:{g}')(x, z_i)

            # upsampling at the end of each scale
            if s < self.num_scales - 1:
                x = scale.get_submodule('upsampling')(x)

        return chunks

    def get_codes(self, x: torch.Tensor):
        codes = self.encode(x, deterministic=True)
        codes = [c.view(x.shape[0], -1) for c in codes]
        return codes

    def decode(self, chunks: list):

        chunks = chunks.copy()

        z_0 = chunks.pop(0)
        b = z_0.shape[0]

        # start from constant prior
        x = self.const_prior.expand(b, -1, -1, -1)

        # first combiner (inject z_0)
        x = self.decoder_combiners.get_submodule('combiner_0:0')(x, z_0)

        for s in range(self.num_scales):

            scale = self.decoder_tower.get_submodule(f'scale_{s}')

            for g in range(self.groups_per_scale[s]):

                if not (s == 0 and g == 0):

                    # group forward
                    group = scale.get_submodule(f'group_{g}')

                    for c in range(self.num_cells_per_group):
                        x = group.get_submodule(f'cell_{c}')(x)

                    z_i = chunks.pop(0)

                    # apply NF
                    if self.use_nf:
                        z_i = self.nf_cells.get_submodule(f'nf_{s}:{g}')(z_i)

                    # combine x and z_i
                    x = self.decoder_combiners.get_submodule(f'combiner_{s}:{g}')(x, z_i)

            # upsampling at the end of each scale
            if s < self.num_scales - 1:
                x = scale.get_submodule('upsampling')(x)

        # postprocessing phase
        x = self.postprocessing_block(x)

        # get logits for mixture
        logits = self.to_logits(x)

        return logits

    def autoencode(self, gt_images: torch.Tensor, deterministic: bool = False):
        """
        :param gt_images: images in 0__1 range
        :return:
        """

        return self.decode(self.encode(gt_images, deterministic))

    def purify(self, batch: torch.Tensor, resample_from: int, temperature: float = 1.):

        b = batch.shape[0]
        assert resample_from > 0

        # preprocessing phase
        normalized_images = (batch * 2) - 1.0  # in range -1 1
        x = self.preprocessing_block(normalized_images)

        # encoding tower phase
        encoder_combiners_x = {}

        for s in range(self.num_scales - 1, -1, -1):

            scale = self.encoder_tower.get_submodule(f'scale_{s}')

            for g in range(self.groups_per_scale[s]):

                group = scale.get_submodule(f'group_{g}')

                for c in range(self.num_cells_per_group):
                    x = group.get_submodule(f'cell_{c}')(x)

                # add intermediate x (will be used as combiner input for this scale)
                if not (s == 0 and g == 0):
                    encoder_combiners_x[f'{s}:{g}'] = x

            if s > 0:
                x = scale.get_submodule(f'downsampling')(x)

        # encoder 0
        x = self.encoder_0(x)

        # obtain q(z_0|x), p(z_0) for KL loss, sample z_0

        # encoder q(z_0|x)
        mu_q, log_sig_q = torch.chunk(self.enc_sampler.get_submodule('sampler_0:0')(x), 2, dim=1)
        dist_enc = Normal(mu_q, log_sig_q)
        z_0, _ = dist_enc.sample()  # uses reparametrization trick

        # apply normalizing flows
        if self.use_nf:
            z_0 = self.nf_cells.get_submodule('nf_0:0')(z_0)

        # decoding phase

        # start from constant prior
        x = self.const_prior.expand(b, -1, -1, -1)

        # first combiner (inject z_0)
        x = self.decoder_combiners.get_submodule('combiner_0:0')(x, z_0)

        count_latents = 1

        for s in range(self.num_scales):

            scale = self.decoder_tower.get_submodule(f'scale_{s}')

            for g in range(self.groups_per_scale[s]):

                if not (s == 0 and g == 0):

                    # group forward
                    group = scale.get_submodule(f'group_{g}')

                    for c in range(self.num_cells_per_group):
                        x = group.get_submodule(f'cell_{c}')(x)

                    if count_latents < resample_from:
                        # obtain q(z_i|x, z_l>i), p(z_i|z_l>i) for KL loss, sample z_i
                        # extract params for p (conditioned on previous decoding)
                        mu_p, log_sig_p = torch.chunk(self.dec_sampler.get_submodule(f'sampler_{s}:{g}')(x), 2, dim=1)

                        # extract params for q (conditioned on encoder features and previous decoding)
                        enc_combiner = self.encoder_combiners.get_submodule(f'combiner_{s}:{g}')
                        enc_sampler = self.enc_sampler.get_submodule(f'sampler_{s}:{g}')
                        mu_q, log_sig_q = torch.chunk(
                            enc_sampler(enc_combiner(encoder_combiners_x[f'{s}:{g}'], x)),
                            2, dim=1)

                        # sample z_i as combination of encoder and decoder params
                        dist_enc = Normal(mu_p + mu_q, log_sig_p + log_sig_q)
                        # z_i, _ = dist_enc.sample()
                        z_i = dist_enc.mu

                    else:

                        # extract params for p (conditioned on previous decoding)
                        mu_p, log_sig_p = torch.chunk(
                            self.dec_sampler.get_submodule(f'sampler_{s}:{g}')(x), 2, dim=1)

                        # sample z_i
                        dist = Normal(mu_p, log_sig_p, temp=temperature)
                        z_i, _ = dist.sample()

                    count_latents += 1

                    # apply NF
                    if self.use_nf:
                        z_i = self.nf_cells.get_submodule(f'nf_{s}:{g}')(z_i)

                    # combine x and z_i
                    x = self.decoder_combiners.get_submodule(f'combiner_{s}:{g}')(x, z_i)

            # upsampling at the end of each scale
            if s < self.num_scales - 1:
                x = scale.get_submodule('upsampling')(x)

        # postprocessing phase
        x = self.postprocessing_block(x)

        # get logits for mixture
        logits = self.to_logits(x)

        return logits
