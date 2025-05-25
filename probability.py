# Copyright 2023 NNAISENSE SA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import functools
from abc import abstractmethod

from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical as torch_Categorical
from torch.distributions.bernoulli import Bernoulli as torch_Bernoulli
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.uniform import Uniform

from math import log

from utils_model import (
    safe_exp,
    safe_log,
    idx_to_float,
    float_to_idx,
    quantize,
    sandwich,
)


class DiscreteDistribution:
    @property
    @abstractmethod
    def probs(self):
        pass

    @functools.cached_property
    def log_probs(self):
        return safe_log(self.probs)

    @functools.cached_property
    def mean(self):
        pass

    @functools.cached_property
    def mode(self):
        pass

    @abstractmethod
    def log_prob(self, x):
        pass

    @abstractmethod
    def sample(self):
        pass


class DiscretizedDistribution(DiscreteDistribution):
    def __init__(self, num_bins, device):
        self.num_bins = num_bins
        self.bin_width = 2.0 / num_bins
        self.half_bin_width = self.bin_width / 2.0
        self.device = device

    @functools.cached_property
    def class_centres(self):
        return torch.arange(
            self.half_bin_width - 1, 1, self.bin_width, device=self.device
        )

    @functools.cached_property
    def class_boundaries(self):
        return torch.arange(
            self.bin_width - 1,
            1 - self.half_bin_width,
            self.bin_width,
            device=self.device,
        )

    @functools.cached_property
    def mean(self):
        return (self.probs * self.class_centres).sum(-1)

    @functools.cached_property
    def mode(self):
        mode_idx = self.probs.argmax(-1).flatten()
        return self.class_centres[mode_idx].reshape(self.probs.shape[:-1])


class Bernoulli(DiscreteDistribution):
    def __init__(self, logits):
        self.bernoulli = torch_Bernoulli(logits=logits, validate_args=False)

    @functools.cached_property
    def probs(self):
        p = self.bernoulli.probs.unsqueeze(-1)
        return torch.cat([1 - p, p], -1)

    @functools.cached_property
    def mode(self):
        return self.bernoulli.mode

    def log_prob(self, x):
        return self.bernoulli.log_prob(x.float())

    def sample(self, sample_shape=torch.Size([])):
        return self.bernoulli.sample(sample_shape)


class DiscreteDistributionFactory:
    @abstractmethod
    def get_dist(
        self, params: torch.Tensor, input_params=None, t=None
    ) -> DiscreteDistribution:
        """Note: input_params and t are only required by PredDistToDataDistFactory."""
        pass


class BernoulliFactory(DiscreteDistributionFactory):
    def get_dist(self, params, input_params=None, t=None):
        return Bernoulli(logits=params.squeeze(-1))


def noise_pred_params_to_data_pred_params(
    noise_pred_params: torch.Tensor,
    input_mean: torch.Tensor,
    t: torch.Tensor,
    min_variance: float,
    min_t=1e-6,
):
    """Convert output parameters that predict the noise added to data, to parameters that predict the data."""
    data_shape = list(noise_pred_params.shape)[:-1]
    noise_pred_params = sandwich(noise_pred_params)
    input_mean = input_mean.flatten(start_dim=1)
    if torch.is_tensor(t):
        t = t.flatten(start_dim=1)
    else:
        t = (input_mean * 0) + t
    alpha_mask = (t < min_t).unsqueeze(-1)
    posterior_var = torch.pow(min_variance, t.clamp(min=min_t))
    gamma = 1 - posterior_var
    A = (input_mean / gamma).unsqueeze(-1)
    B = (posterior_var / gamma).sqrt().unsqueeze(-1)
    data_pred_params = []
    if noise_pred_params.size(-1) == 1:
        noise_pred_mean = noise_pred_params
    elif noise_pred_params.size(-1) == 2:
        noise_pred_mean, noise_pred_log_dev = noise_pred_params.chunk(2, -1)
    else:
        assert noise_pred_params.size(-1) % 3 == 0
        mix_wt_logits, noise_pred_mean, noise_pred_log_dev = noise_pred_params.chunk(
            3, -1
        )
        data_pred_params.append(mix_wt_logits)
    data_pred_mean = A - (B * noise_pred_mean)
    data_pred_mean = torch.where(alpha_mask, 0 * data_pred_mean, data_pred_mean)
    data_pred_params.append(data_pred_mean)
    if noise_pred_params.size(-1) >= 2:
        noise_pred_dev = safe_exp(noise_pred_log_dev)
        data_pred_dev = B * noise_pred_dev
        data_pred_dev = torch.where(alpha_mask, 1 + (0 * data_pred_dev), data_pred_dev)
        data_pred_params.append(data_pred_dev)
    data_pred_params = torch.cat(data_pred_params, -1)
    data_pred_params = data_pred_params.reshape(data_shape + [-1])
    return data_pred_params


class PredDistToDataDistFactory(DiscreteDistributionFactory):
    def __init__(self, data_dist_factory, min_variance, min_t=1e-6):
        self.data_dist_factory = data_dist_factory
        self.data_dist_factory.log_dev = False
        self.min_variance = min_variance
        self.min_t = min_t

    def get_dist(self, params, input_params, t):
        data_pred_params = noise_pred_params_to_data_pred_params(
            params, input_params[0], t, self.min_variance, self.min_t
        )
        return self.data_dist_factory.get_dist(data_pred_params)
