import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils
# from piq import ssim,psnr,LPIPS
import logging
import time
import math
from typing import Optional, Sequence


class DeNormalize(nn.Module):               
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        """
        Args:
            tensor (Tensor): Normalized Tensor image.

        Returns:
            Tensor: Denormalized Tensor.
        """
        return self._denormalize(tensor)

    def _denormalize(self, tensor):
        tensor = tensor.clone()

        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        if (std == 0).any():
            raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        # tensor.sub_(mean).div_(std)
        tensor.mul_(std).add_(mean)
        return tensor
    
# 分割cifar10为两半
def split_data(data, dataset):
    if dataset == 'cifar10':
        x_a = data[:, :, :, 0:16]
        x_b = data[:, :, :, 16:32]
    elif dataset == 'tinyImagenet':
        x_a = data[:, :, :, 0:32]
        x_b = data[:, :, :, 32:64]
    elif dataset == 'bank':
        x_a = data[:, 0:10]
        x_b = data[:, 10:20]
    elif dataset == 'drive':
        x_a = data[:, 0:24]
        x_b = data[:, 24:48]
    return x_a, x_b



def gradient_penalty(discriminator, x, x_gen,device):
    """
    Args:
        x
        x_gen

    Returns:
        d_regularizer
    """
    epsilon = torch.rand([x.shape[0], 1, 1, 1]).to(device)
    x_hat = epsilon * x + (1 - epsilon) * x_gen
    x_hat = torch.autograd.Variable(x_hat, requires_grad=True)
    from torch.autograd import grad
    d_hat = discriminator(x_hat)
    gradients = grad(outputs=d_hat, inputs=x_hat,
                    grad_outputs=torch.ones_like(d_hat).to(device),
                    retain_graph=True, create_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0),  -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1)**2).mean()

    return penalty


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dm %ds' % (m, s) if h==0 else '%dh %dm %ds' % (h, m, s)

def timeSince(since):
    now = time.time()
    s = now - since
    return '%s' % (asMinutes(s))



class CorrelationAlignmentLoss(nn.Module):
    r"""The `Correlation Alignment Loss` in
    `Deep CORAL: Correlation Alignment for Deep Domain Adaptation (ECCV 2016) <https://arxiv.org/pdf/1607.01719.pdf>`_.

    Given source features :math:`f_S` and target features :math:`f_T`, the covariance matrices are given by

    .. math::
        C_S = \frac{1}{n_S-1}(f_S^Tf_S-\frac{1}{n_S}(\textbf{1}^Tf_S)^T(\textbf{1}^Tf_S))
    .. math::
        C_T = \frac{1}{n_T-1}(f_T^Tf_T-\frac{1}{n_T}(\textbf{1}^Tf_T)^T(\textbf{1}^Tf_T))

    where :math:`\textbf{1}` denotes a column vector with all elements equal to 1, :math:`n_S, n_T` denotes number of
    source and target samples, respectively. We use :math:`d` to denote feature dimension, use
    :math:`{\Vert\cdot\Vert}^2_F` to denote the squared matrix `Frobenius norm`. The correlation alignment loss is
    given by

    .. math::
        l_{CORAL} = \frac{1}{4d^2}\Vert C_S-C_T \Vert^2_F

    Inputs:
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - f_t (tensor): feature representations on target domain, :math:`f^t`

    Shape:
        - f_s, f_t: :math:`(N, d)` where d means the dimension of input features, :math:`N=n_S=n_T` is mini-batch size.
        - Outputs: scalar.
    """

    def __init__(self):
        super(CorrelationAlignmentLoss, self).__init__()

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        mean_s = f_s.mean(0, keepdim=True)
        mean_t = f_t.mean(0, keepdim=True)
        cent_s = f_s - mean_s
        cent_t = f_t - mean_t
        cov_s = torch.mm(cent_s.t(), cent_s) / (len(f_s) - 1)
        cov_t = torch.mm(cent_t.t(), cent_t) / (len(f_t) - 1)

        mean_diff = (mean_s - mean_t).pow(2).mean()
        cov_diff = (cov_s - cov_t).pow(2).mean()

        return mean_diff + cov_diff


class MultipleKernelMaximumMeanDiscrepancy(nn.Module):
    """
    @acknowledgment:This code is based on the a publicly available code repository.<https://github.com/thuml/Transfer-Learning-Library>
    @author: Junguang Jiang
    """
    r"""The Multiple Kernel Maximum Mean Discrepancy (MK-MMD) used in
    `Learning Transferable Features with Deep Adaptation Networks (ICML 2015) <https://arxiv.org/pdf/1502.02791>`_

    Given source domain :math:`\mathcal{D}_s` of :math:`n_s` labeled points and target domain :math:`\mathcal{D}_t`
    of :math:`n_t` unlabeled points drawn i.i.d. from P and Q respectively, the deep networks will generate
    activations as :math:`\{z_i^s\}_{i=1}^{n_s}` and :math:`\{z_i^t\}_{i=1}^{n_t}`.
    The MK-MMD :math:`D_k (P, Q)` between probability distributions P and Q is defined as

    .. math::
        D_k(P, Q) \triangleq \| E_p [\phi(z^s)] - E_q [\phi(z^t)] \|^2_{\mathcal{H}_k},

    :math:`k` is a kernel function in the function space

    .. math::
        \mathcal{K} \triangleq \{ k=\sum_{u=1}^{m}\beta_{u} k_{u} \}

    where :math:`k_{u}` is a single kernel.

    Using kernel trick, MK-MMD can be computed as

    .. math::
        \hat{D}_k(P, Q) &=
        \dfrac{1}{n_s^2} \sum_{i=1}^{n_s}\sum_{j=1}^{n_s} k(z_i^{s}, z_j^{s})\\
        &+ \dfrac{1}{n_t^2} \sum_{i=1}^{n_t}\sum_{j=1}^{n_t} k(z_i^{t}, z_j^{t})\\
        &- \dfrac{2}{n_s n_t} \sum_{i=1}^{n_s}\sum_{j=1}^{n_t} k(z_i^{s}, z_j^{t}).\\

    Args:
        kernels (tuple(torch.nn.Module)): kernel functions.
        linear (bool): whether use the linear version of DAN. Default: False

    Inputs:
        - z_s (tensor): activations from the source domain, :math:`z^s`
        - z_t (tensor): activations from the target domain, :math:`z^t`

    Shape:
        - Inputs: :math:`(minibatch, *)`  where * means any dimension
        - Outputs: scalar

    .. note::
        Activations :math:`z^{s}` and :math:`z^{t}` must have the same shape.

    .. note::
        The kernel values will add up when there are multiple kernels.

    Examples::

        >>> from tllib.modules.kernels import GaussianKernel
        >>> feature_dim = 1024
        >>> batch_size = 10
        >>> kernels = (GaussianKernel(alpha=0.5), GaussianKernel(alpha=1.), GaussianKernel(alpha=2.))
        >>> loss = MultipleKernelMaximumMeanDiscrepancy(kernels)
        >>> # features from source domain and target domain
        >>> z_s, z_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> output = loss(z_s, z_t)
    """

    def __init__(self, kernels: Sequence[nn.Module], linear: Optional[bool] = False):
        super(MultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        features = torch.cat([z_s, z_t], dim=0)
        batch_size = int(z_s.size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s.device)
        # print(self.index_matrix.shape)
        # exit()


        kernel_matrix = sum([kernel(features) for kernel in self.kernels])  # Add up the matrix of each kernel
        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        # print(self.kernels[0](features).shape)
        # exit()

        loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)

        return loss


def _update_index_matrix(batch_size: int, index_matrix: Optional[torch.Tensor] = None,
                         linear: Optional[bool] = True) -> torch.Tensor:
    r"""
    Update the `index_matrix` which convert `kernel_matrix` to loss.
    If `index_matrix` is a tensor with shape (2 x batch_size, 2 x batch_size), then return `index_matrix`.
    Else return a new tensor with shape (2 x batch_size, 2 x batch_size).
    """
    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
        if linear:
            for i in range(batch_size):
                s1, s2 = i, (i + 1) % batch_size
                t1, t2 = s1 + batch_size, s2 + batch_size
                index_matrix[s1, s2] = 1. / float(batch_size)
                index_matrix[t1, t2] = 1. / float(batch_size)
                index_matrix[s1, t2] = -1. / float(batch_size)
                index_matrix[s2, t1] = -1. / float(batch_size)
        else:
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
                        index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
            for i in range(batch_size):
                for j in range(batch_size):
                    index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
                    index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)
    return index_matrix
class GaussianKernel(nn.Module):
    r"""Gaussian Kernel Matrix

    Gaussian Kernel k is defined by

    .. math::
        k(x_1, x_2) = \exp \left( - \dfrac{\| x_1 - x_2 \|^2}{2\sigma^2} \right)

    where :math:`x_1, x_2 \in R^d` are 1-d tensors.

    Gaussian Kernel Matrix K is defined on input group :math:`X=(x_1, x_2, ..., x_m),`

    .. math::
        K(X)_{i,j} = k(x_i, x_j)

    Also by default, during training this layer keeps running estimates of the
    mean of L2 distances, which are then used to set hyperparameter  :math:`\sigma`.
    Mathematically, the estimation is :math:`\sigma^2 = \dfrac{\alpha}{n^2}\sum_{i,j} \| x_i - x_j \|^2`.
    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and use a fixed :math:`\sigma` instead.

    Args:
        sigma (float, optional): bandwidth :math:`\sigma`. Default: None
        track_running_stats (bool, optional): If ``True``, this module tracks the running mean of :math:`\sigma^2`.
          Otherwise, it won't track such statistics and always uses fix :math:`\sigma^2`. Default: ``True``
        alpha (float, optional): :math:`\alpha` which decides the magnitude of :math:`\sigma^2` when track_running_stats is set to ``True``

    Inputs:
        - X (tensor): input group :math:`X`

    Shape:
        - Inputs: :math:`(minibatch, F)` where F means the dimension of input features.
        - Outputs: :math:`(minibatch, minibatch)`
    """

    def __init__(self, sigma: Optional[float] = None, track_running_stats: Optional[bool] = True,
                 alpha: Optional[float] = 1.):
        super(GaussianKernel, self).__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = torch.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)

        if self.track_running_stats:
            self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())

        return torch.exp(-l2_distance_square / (2 * self.sigma_square))