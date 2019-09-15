import itertools as it

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import vel.util.network as net_util

from vel.api import GradientModel, ModelFactory
from vel.metric import AveragingNamedMetric
from vel.metric.loss_metric import Loss
from vel.module.layers import Flatten, Reshape


class MnistCnnVAE(GradientModel):
    """
    A simple MNIST variational autoencoder, containing 3 convolutional layers.
    """

    def __init__(self, img_rows, img_cols, img_channels, channels=None, representation_length=32):
        super(MnistCnnVAE, self).__init__()

        if channels is None:
            channels = [16, 32, 32]

        layer_series = [
            (3, 1, 1),
            (3, 1, 2),
            (3, 1, 2),
        ]

        self.representation_length = representation_length

        self.final_width = net_util.convolutional_layer_series(img_rows, layer_series)
        self.final_height = net_util.convolutional_layer_series(img_cols, layer_series)
        self.channels = channels

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=img_channels, out_channels=channels[0], kernel_size=(3, 3), padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=(3, 3), stride=2, padding=1),
            Flatten(),
            nn.Linear(self.final_width * self.final_height * channels[2], representation_length * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(representation_length, self.final_width * self.final_height * channels[2]),
            nn.ReLU(True),
            Reshape(channels[2], self.final_width, self.final_height),
            nn.ConvTranspose2d(
                in_channels=channels[2], out_channels=channels[1], kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=channels[1], out_channels=channels[0], kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=channels[0], out_channels=img_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    @staticmethod
    def _weight_initializer(tensor):
        init.xavier_uniform_(tensor.weight, gain=init.calculate_gain('relu'))
        init.constant_(tensor.bias, 0.0)

    def reset_weights(self):
        for m in it.chain(self.encoder, self.decoder):
            if isinstance(m, nn.Conv2d):
                self._weight_initializer(m)
            elif isinstance(m, nn.ConvTranspose2d):
                self._weight_initializer(m)
            elif isinstance(m, nn.Linear):
                self._weight_initializer(m)

    def encoder_distribution(self, sample):
        encoding = self.encoder(sample)
        mu = encoding[:, :self.representation_length]
        # I encode std directly as a softplus, rather than exp(logstd)
        std = F.softplus(encoding[:, self.representation_length:])

        return mu, std

    def encode(self, sample):
        mu, std = self.encoder_distribution(sample)
        # Sample z
        return mu + torch.randn_like(std) * std

    def decode(self, sample):
        # We don't sample here, because decoder is so weak it doesn't make sense
        return self.decoder(sample)

    def forward(self, sample):
        mu, std = self.encoder_distribution(sample)

        # Sample z
        z = mu + torch.randn_like(std) * std
        decoded = self.decoder(z)

        return {
            'decoded': decoded,
            'encoding': z,
            'mu': mu,
            'std': std
        }

    def calculate_gradient(self, data):
        """ Calculate a gradient of loss function """
        output = self(data['x'])

        # ELBO is E_q log p(x, z) / q(z | x)
        # Which can be expressed in many equivalent forms:
        # (1) E_q log p(x | z) + log p(z) - log q(z | x)
        # (2) E_q log p(x | z) - D_KL(p(z) || q(z | x))
        # (3) E_q log p(x) - D_KL(p(z | x) || q(z | x)Biblio)

        # Form 3 is interesting from a theoretical standpoint, but is intractable to compute directly
        # While forms (1) and (2) can be computed directly.
        # Positive aspect of form (2) is that KL divergence can be calculated analytically
        # further reducing the variance of the gradient

        y_pred = output['decoded']

        mu = output['mu']
        std = output['std']
        var = std ** 2

        # Analytical solution of KL divergence
        kl_divergence = - 0.5 * (1 + torch.log(var) - mu ** 2 - var).sum(dim=1)
        kl_divergence = kl_divergence.mean()

        # Diag-gaussian likelihood
        # likelihood = 0.5 * F.mse_loss(y_pred, y_true)

        # We must sum over all image axis and average only on minibatch axis
        # Log prob p(x | z) in the case where the output distribution is Bernoulli(p)
        likelihood = F.binary_cross_entropy(y_pred, data['y'], reduction='none').sum((1, 2, 3)).mean()

        elbo = likelihood + kl_divergence

        nll = self.nll(data['x'], num_posterior_samples=5)

        if self.training:
            elbo.backward()

        return {
            'loss': elbo.item(),
            'nll': nll.mean().item(),
            'reconstruction': likelihood.item(),
            'kl_divergence': kl_divergence.item()
        }

    def logmeanexp(self, inputs, dim=1):
        if inputs.size(dim) == 1:
            return inputs
        else:
            input_max = inputs.max(dim, keepdim=True)[0]
            return (inputs - input_max).exp().mean(dim).log() + input_max.squeeze(dim=dim)

    @torch.no_grad()
    def nll(self, data_sample, num_posterior_samples: int = 1):
        """
        Upper bound on negative log-likelihood of supplied data.
        If num samples goes to infinity, the nll of data should
        approach true value
        """
        assert num_posterior_samples >= 1, "Need at least one posterior sample"

        buffer = []

        mu, std = self.encoder_distribution(data_sample)
        var = std ** 2

        kl_divergence = - 0.5 * (1 + torch.log(var) - mu ** 2 - var).sum(dim=1)

        for i in range(num_posterior_samples):
            z = mu + torch.randn_like(std) * std
            y_pred = self.decoder(z)

            likelihood = F.binary_cross_entropy(y_pred, data_sample, reduction='none').sum((1, 2, 3))
            elbo = likelihood + kl_divergence

            buffer.append(-elbo)

        averaged = self.logmeanexp(torch.stack(buffer, dim=-1), dim=-1)
        return -averaged

    def metrics(self):
        """ Set of metrics for this model """
        return [
            Loss(),
            AveragingNamedMetric('reconstruction', scope="train"),
            AveragingNamedMetric('kl_divergence', scope="train")
        ]


def create(img_rows, img_cols, img_channels, channels=None, representation_length=32):
    """ Vel factory function """
    if channels is None:
        channels = [16, 32, 32]

    def instantiate(**_):
        return MnistCnnVAE(
            img_rows, img_cols, img_channels, channels=channels, representation_length=representation_length
        )

    return ModelFactory.generic(instantiate)
