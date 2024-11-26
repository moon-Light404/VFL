import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils
# from piq import ssim,psnr,LPIPS
import logging
import time
import math


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
    else:
        x_a = data[:, 0:10]
        x_b = data[:, 10:20]
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
        # conv_s、conv_t: covariance matrix 都是d * d矩阵
        cov_s = torch.mm(cent_s.t(), cent_s) / (len(f_s) - 1)
        cov_t = torch.mm(cent_t.t(), cent_t) / (len(f_t) - 1)

        mean_diff = (mean_s - mean_t).pow(2).mean()
        cov_diff = (cov_s - cov_t).pow(2).mean()

        return mean_diff + cov_diff


