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


