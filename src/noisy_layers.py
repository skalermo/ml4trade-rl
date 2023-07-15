import math
import torch as th
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class NoisyLinear(nn.Module):
    r"""Applies a noisy linear transformation to the incoming data.
    During training:
        .. math:: `y = (mu_w + sigma_w \cdot epsilon_w)x
            + mu_b + sigma_b \cdot epsilon_b`
    During evaluation:
        .. math:: `y = mu_w * x + mu_b`
    More details can be found in the paper `Noisy Networks for Exploration` _ .
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: True
        factorized: whether or not to use factorized noise.
            Default: True
        std_init: constant for weight_sigma and bias_sigma initialization.
            If None, defaults to 0.017 for independent and 0.4 for factorized.
            Default: None
    Shape:
        - Input: :math:`(N, in\_features)`
        - Output: :math:`(N, out\_features)`
    Attributes:
        weight: the learnable weights of the module of shape
            (out_features x in_features)
        bias:   the learnable bias of the module of shape (out_features)
    Methods:
        resample: resamples the noise tensors
    Examples::
        >>> m = nn.NoisyLinear(20, 30)
        >>> input = autograd.Variable(torch.randn(128, 20))
        >>> output = m(input)
        >>> m.resample()
        >>> output_new = m(input)
        >>> print(output)
        >>> print(output_new)
    """
    def __init__(self, in_features, out_features, bias=True, factorized=False, std_init=None):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.factorized = factorized
        self.include_bias = bias
        self.weight_mu = nn.Parameter(th.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(th.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', th.Tensor(out_features, in_features))
        if self.include_bias:
            self.bias_mu = nn.Parameter(th.Tensor(out_features))
            self.bias_sigma = nn.Parameter(th.Tensor(out_features))
            self.register_buffer('bias_epsilon', th.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        if not std_init:
            if self.factorized:
                self.std_init = 0.4
            else:
                self.std_init = 0.017
        else:
            self.std_init = std_init
        self.reset_parameters()
        self.resample()

    def reset_parameters(self):
        if self.factorized:
            mu_range = 1. / math.sqrt(self.weight_mu.size(1))
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
            if self.include_bias:
                self.bias_mu.data.uniform_(-mu_range, mu_range)
                self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
        else:
            mu_range = math.sqrt(3. / self.weight_mu.size(1))
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std_init)
        if self.include_bias:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.std_init)

    def _scale_noise(self, size):
        x = th.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

    def resample(self):
        if self.factorized:
            epsilon_in = self._scale_noise(self.in_features)
            epsilon_out = self._scale_noise(self.out_features)
            self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
            if self.include_bias:
                self.bias_epsilon.copy_(self._scale_noise(self.out_features))
        else:
            self.weight_epsilon.normal_()
            if self.include_bias:
                self.bias_epsilon.normal_()

    def forward(self, input):
        if self.training:
            return F.linear(input,
                            self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon)),
                            self.bias_mu + self.bias_sigma.mul(Variable(self.bias_epsilon)))
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ', Factorized: ' \
               + str(self.factorized) + ')'

