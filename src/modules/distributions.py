from einops import rearrange, repeat, pack
import numpy as np
import torch


def gumbel_sampling(logits: torch.Tensor, temperature: float = 1.0):
    """
    better that categorical sampling
    """

    one_hot_probs = torch.zeros_like(logits)

    gumbel_noise = torch.zeros_like(logits).uniform_(1e-5, 1. - 1e-5)
    gumbel_noise = -torch.log(-torch.log(gumbel_noise))
    logits = torch.argmax(logits / temperature + gumbel_noise, 1, keepdim=True)

    return one_hot_probs.scatter_(1, logits, 1)


def soft_clamp(x: torch.Tensor, n: float = 5.):
    """
    check https://github.com/NVlabs/NVAE/issues/27

    Clamp values in a differentiable and soft way between [-n; n]: n. * torch.tanh(x / n.)
    This helps a little in maintaining small KL terms.

    Note: why in the source code they used 5. it is not explained.
    """
    return x.div(n).tanh_().mul(n)


class Normal:
    def __init__(self, mu, log_sigma, temp=1.):
        self.mu = soft_clamp(mu)
        self.sigma = temp * torch.exp(soft_clamp(log_sigma))  # removed extra 1e-2 term

    def sample(self):
        """
        sample new z using reparametrization trick z = mu + eps * sigma
            eps is sampled from normal (has the same shape as mu, sigma)
        return z, epsilon
        """
        eps = torch.zeros_like(self.mu).normal_()
        z = eps.mul_(self.sigma).add_(self.mu)
        return z, eps

    def sample_given_eps(self, eps):
        return self.mu + eps * self.sigma

    def log_p(self, samples):
        """ compute the log likelihood of normal probability on samples. """
        normalized_samples = (samples - self.mu) / self.sigma
        log_p = - 0.5 * torch.square(normalized_samples) - 0.5 * np.log(2 * np.pi) - torch.log(self.sigma)
        return log_p

    def kl(self, prior):
        """
        check section 3.2 of the paper, paragraph: Residual Normal Distributions
        Here self.mu, self.sigma are the parameters of the posterior distribution
        prior.mu, prior.sigma are the parameters of the prior distribution

        p(z_i|z_{l<i}) is the prior, defined as N(μ_p, σ_p) where both params are conditioned on all z_{l<i}

        q(z_i|z_{l<i}, x) is the distribution from the encoder (self), defined as:
            q = N(μ_p + Δμ_q, σ_p * Δσ_q), where Δμ_q, Δσ_q are the relative shift and scale given by the hierarchical
            nature of the distribution.

        So basically, the self.mu and self.sigma are the parameters of the posterior:
            self.mu = μ_p + Δμ_q
            self.sigma = σ_p * Δσ_q

        The KL Loss between two normal distributions a = N(μ_1, σ_1), b = N(μ_2, σ_2) is given by:
            0.5 [ (μ_2 - μ_1)**2 / σ_2**2 ] + 0.5 (σ_1**2 / σ_2**2) - 0.5 [ln(σ_1**2 / σ_2**2)] - 0.5

        proof: https://statproofbook.github.io/P/norm-kl.html

        In our case: μ_1 = self.mu; μ_2 = prior.mu; σ_1 = self.sigma; σ_2 = prior.sigma

        So the three terms in the formula above become:
            1. 0.5 [ (μ_p - μ_p + Δμ_q)**2 / σ_p**2] =  0.5 [ Δμ_q**2 / σ_p**2]
            2. 0.5 ((σ_p * Δσ_q)**2 / σ_p**2) = 0.5 [Δσ_q**2]
            3. 0.5 [ln((σ_p * Δσ_q)**2 / σ_p**2)] = 0.5 ln(Δσ_q**2)

        The final formula is thus the one written in Equation 2 and:
        Δμ_q = self.mu - prior.mu
        Δσ_q = self.sigma / prior.sigma

        """

        delta_mu = self.mu - prior.mu
        delta_sigma = self.sigma / prior.sigma
        first_term = torch.pow(delta_mu, 2) / torch.pow(prior.sigma, 2)
        kl = 0.5 * (first_term + torch.pow(delta_sigma, 2)) - 0.5 - torch.log(delta_sigma)
        return kl


class DiscMixLogistic:
    """
    code adapted from https://github.com/mgp123/nvae/blob/main/model/distributions.py
    and from https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py
    """

    def __init__(self, params: torch.Tensor, img_channels: int = 3, num_bits: int = 8):
        """
        Defines 3 params for N Logistics: 1. which to take, 2. mu, 3. log_scale
        Additionally dimension is the final logits probabilities
        Args:
            params: B, N_MIX + N_MIX * 3 * C, H, W
            num_bits: final image range goes from [0 to 2. ** num_bits -1)
        """

        if img_channels != 3:
            raise NotImplementedError("actually works only with 3D images")

        # hyperparams
        self.img_channels = img_channels
        self.b, x, self.h, self.w = params.shape
        self.num_mixtures = x // (1 + img_channels * 3)
        self.max_val = 2. ** num_bits - 1

        self.logits = rearrange(params[:, :self.num_mixtures], 'b n h w -> b n (h w)')

        # get params for logistics means, log_scales and coefficients
        params = rearrange(params[:, self.num_mixtures:], 'b (n c) h w -> b n c (h w)', n=self.num_mixtures)
        m, s, k = torch.chunk(params, chunks=3, dim=2)

        self.means_logits = m
        self.log_scales_logits = torch.clamp(s, min=-7.0)
        self.logistic_coefficients = torch.tanh(k)  # [-1, 1.] range

    def log_prob(self, samples: torch.Tensor):
        """
        Args:
            samples: ground truth images in -1_1 range.
        Returns:

        """

        b, _, h, w = samples.shape

        # expand with num mixtures
        samples = repeat(samples, 'b c h w -> b n c (h w)', n=self.num_mixtures)

        # adjust means based on preceding channels (check section 2.2 of PixelCNN++ paper)
        # https://openreview.net/pdf?id=BJrFC6ceg
        r_mean = self.means_logits[:, :, 0, :].unsqueeze(2)  # B, N, RED_CH, (H, W)

        # B, N, GREEN_CH, (H, W)
        g_mean = self.means_logits[:, :, 1, :].unsqueeze(2)
        g_mean = g_mean + (self.logistic_coefficients[:, :, 0, :] * samples[:, :, 0, :]).unsqueeze(2)

        # B, N, BLUE_CH, (H, W)
        b_mean = self.means_logits[:, :, 2, :].unsqueeze(2)
        b_mean = b_mean + (self.logistic_coefficients[:, :, 1, :] * samples[:, :, 0, :]).unsqueeze(2)
        b_mean = b_mean + (self.logistic_coefficients[:, :, 2, :] * samples[:, :, 1, :]).unsqueeze(2)

        adjusted_means, _ = pack([r_mean, g_mean, b_mean], 'b n * d')  # B N C (H W)

        # compute CDF in the neighborhood of each sample
        centered = samples - adjusted_means
        neg_scale = torch.exp(- self.log_scales_logits)

        plus_in = neg_scale * (centered + 1. / self.max_val)
        cdf_plus = torch.sigmoid(plus_in)

        min_in = neg_scale * (centered - 1. / self.max_val)
        cdf_min = torch.sigmoid(min_in)

        # log probability for edge case of 0
        log_cdf_plus = plus_in - torch.nn.functional.softplus(plus_in)

        # log probability for edge case of 255
        log_one_minus_cdf_min = - torch.nn.functional.softplus(min_in)

        # probability for all other cases
        cdf_delta = cdf_plus - cdf_min

        # probability for very low values (avoid nan in following computations)
        safe_log_probs = neg_scale * centered
        safe_log_probs = safe_log_probs - self.log_scales_logits - 2. * torch.nn.functional.softplus(safe_log_probs)
        safe_log_probs = safe_log_probs - np.log(self.max_val / 2)

        # select the right output: left edge case, right edge case, normal case (filtered for low values)
        safe_log_probs = torch.where(cdf_delta > 1e-5,
                                     torch.log(torch.clamp(cdf_delta, min=1e-10)),
                                     safe_log_probs)

        log_probs = torch.where(samples < -0.999,
                                log_cdf_plus,
                                torch.where(samples > 0.99,
                                            log_one_minus_cdf_min,
                                            safe_log_probs)
                                )

        # log_probs is B N C (H W) --> sum over channels
        # logits is B N (H W)
        log_probs = torch.sum(log_probs, dim=2) + torch.nn.functional.log_softmax(self.logits, dim=1)
        return torch.logsumexp(log_probs, dim=1)

    def sample(self):

        # Gumbel Sampling of shape B, (H W) of indices in range 0 __ N_MIXTURES
        logistic_selection_mask = gumbel_sampling(self.logits).unsqueeze(2)

        # select parameters and sum on NUM_MIXTURES --> B, C, (H W)
        selected_mu = torch.sum(self.means_logits * logistic_selection_mask, dim=1)
        selected_scale = torch.sum(self.log_scales_logits * logistic_selection_mask, dim=1)
        selected_k = torch.sum(self.logistic_coefficients * logistic_selection_mask, dim=1)

        # sample from logistic with default parameters and then scale
        # pytorch example for Logistic Distribution --> https://pytorch.org/docs/stable/distributions.html
        # why the logistic distribution does not exist in pytorch? --> https://github.com/pytorch/pytorch/issues/7857
        base_sample = torch.zeros_like(selected_mu).uniform_(1e-5, 1. - 1e-5)  # avoiding corner values
        base_sample = torch.log(base_sample) - torch.log(1 - base_sample)

        x = selected_mu + torch.exp(selected_scale) * base_sample  # B, C, (H W)

        # adjust given previous channels (clamp in -1, 1. range)
        r_x = torch.clamp(x[:, 0, :], -1., 1.)  # B, (H, W)

        g_x = x[:, 1, :] + selected_k[:, 0, :] * r_x
        g_x = torch.clamp(g_x, -1., 1.)  # B, (H, W)

        b_x = x[:, 2, :] + selected_k[:, 1, :] * r_x + selected_k[:, 2, :] * g_x
        b_x = torch.clamp(b_x, -1., 1.)  # B, (H, W)

        # get final B 3 (H W)
        x = pack([r_x.unsqueeze(1), g_x.unsqueeze(1), b_x.unsqueeze(1)], 'b * d')[0]
        return rearrange(x, 'b c (h w) -> b c h w', h=self.h, w=self.w)

    def mean(self):

        # just scale means by their corresponding probs
        probs = torch.softmax(self.logits, dim=1).unsqueeze(2)

        # select logistic parameters
        selected_mu = torch.sum(self.means_logits * probs, dim=1)  # B, 3, (H, W)
        selected_k = torch.sum(self.logistic_coefficients * probs, dim=1)  # B, 3, (H, W)

        # don't sample, use mean
        x = selected_mu

        # adjust given previous channels (clamp in -1, 1. range)
        r_x = torch.clamp(x[:, 0, :], -1., 1.)  # B, (H, W)

        g_x = x[:, 1, :] + selected_k[:, 0, :] * r_x
        g_x = torch.clamp(g_x, -1., 1.)  # B, (H, W)

        b_x = x[:, 2, :] + selected_k[:, 1, :] * r_x + selected_k[:, 2, :] * g_x
        b_x = torch.clamp(b_x, -1., 1.)  # B, (H, W)

        # get final B 3 (H W)
        x = pack([r_x.unsqueeze(1), g_x.unsqueeze(1), b_x.unsqueeze(1)], 'b * d')[0]
        return rearrange(x, 'b c (h w) -> b c h w', h=self.h, w=self.w)
