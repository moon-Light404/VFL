import argparse
import os
from typing import Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F_1
import copy
import numpy as np
import torchvision.utils as vutils
# from piq import ssim,psnr,LPIPS
import logging
from utils import split_data, gradient_penalty, DeNormalize


def pseudo_training(target_vflnn, pseudo_model, pseudo_inverse_model, pseudo_optimizer, pseudo_inverse_optimizer, discriminator, discriminator_optimizer, target_data, target_label, shadow_data, shadow_label, device, n, cat_dimension, args):
    
    # 正常VFL训练
    target_vflnn.train()

    target_data = target_data.to(device)
    target_label = target_label.to(device)

    target_vflnn.zero_grads()
    # 分割数据
    target_x_a, target_x_b = split_data(target_data, args.dataset)
    # 训练VFL模型
    target_vflnn_output = target_vflnn(target_x_a, target_x_b)
    # passive party的中间特征
    target_vflnn_pas_intermediate = target_vflnn.intermediate_to_server1.detach()
    # active party的中间特征
    target_vflnn_act_intermediate = target_vflnn.intermediate_to_server2.detach()

    target_vflnn_loss = F.cross_entropy(target_vflnn_output, target_label)
    
    target_vflnn_loss.backward()

    target_vflnn.backward()
    # 更新整个vfl模型
    target_vflnn.step()
    

    # 整个VFL模型不更新
    for para in target_vflnn.client1.parameters():
        para.requires_grad = False
    for para in target_vflnn.server.parameters():
        para.requires_grad = False  

    
    # 训练伪模型，比较的特征空间是图像的一半
    pseudo_model.train()
    pseudo_inverse_model.train()
    discriminator.train()

    shadow_data = shadow_data.to(device)
    shadow_label = shadow_label.to(device)
    # 辅助数据分成两半，一般给伪模型，一般给主动方的底部模型fa
    shadow_x_a, shadow_x_b = split_data(shadow_data, args.dataset)


    pseudo_optimizer.zero_grad()
    pseduo_output = pseudo_model(shadow_x_a) # 伪模型的特征空间
    
    # 把伪模型的特征空间输入到鉴别器中
    d_output_pseudo = discriminator(pseduo_output)
    pseudo_loss = torch.mean(d_output_pseudo)

    pseudo_loss.backward()
    pseudo_optimizer.step()
    # if args.if_update == False:
    #     for para in pseudo_model.parameters():
    #         para.requires_grad = False

    # 训练逆网络，同时优化伪模型和fa
    pseudo_inverse_optimizer.zero_grad()
    target_vflnn.client2_optimizer.zero_grad()
    if args.if_update == True:
        pseudo_model.zero_grad()

    with torch.no_grad():
        # pseudo_model 是client1的伪模型，伪模型和fa都更新
        pseudo_inverse_input_a = pseudo_model(shadow_x_a)
        pseduo_inverse_input_b = target_vflnn.client2(shadow_x_b)
    # 两个特征拼接输入到逆网络，恢复数据 
    pseudo_inverse_input = torch.cat((pseudo_inverse_input_a, pseduo_inverse_input_b), cat_dimension)
    pseudo_inverse_output = pseudo_inverse_model(pseudo_inverse_input)
    pseudo_inverse_loss = F.mse_loss(pseudo_inverse_output, shadow_data)
    # 更新逆网络、伪模型、fa
    pseudo_inverse_loss.backward()
    pseudo_inverse_optimizer.step()
    # 更新恶意方的底部模型
    target_vflnn.client2_optimizer.step()
    # if_update更新伪模型
    if args.if_update == True:
        pseudo_optimizer.step()

    # 进一步更新伪模型，只更新伪模型
    for para in target_vflnn.client2.parameters():
        para.requires_grad = False
    # 学习服务器的知识
    loss_flag = 1000
    with torch.no_grad():
        vflnn_output = target_vflnn(target_x_a, target_x_b)
        target_vflnn_loss = F.cross_entropy(vflnn_output, target_label)
        loss_flag = target_vflnn_loss.clone().item()
    # loss_flag < 0 就不进入这个分支
    if args.loss_threshold > 0 and loss_flag < args.loss_threshold:
        target_vflnn.zero_grads()
        pseudo_optimizer.zero_grad()
        target_vflnn.eval()
        classifier_input1 = pseudo_model(shadow_x_a)
        classifier_input2 = target_vflnn.client2(shadow_x_b).detach()
        classifer_output = target_vflnn.server([classifier_input1, classifier_input2])
        classifier_loss = F.cross_entropy(classifer_output, shadow_label)
        classifier_loss.backward()
        pseudo_optimizer.step() # 更新伪模型,更接近被动方模型


    # 更新鉴别器，此时不能更新伪模型，设为detach()
    discriminator_optimizer.zero_grad()
    
    pseduo_output_ = pseduo_output.detach()
    # 目标客户端特征空间
    target_vflnn_pas_intermediate_ = target_vflnn_pas_intermediate.detach()
     
    adv_target_logits = discriminator(target_vflnn_pas_intermediate_)
    adv_pseudo_logits = discriminator(pseduo_output_)
    loss_discr_true = torch.mean(adv_target_logits)
    loss_discr_fake = -torch.mean(adv_pseudo_logits)
    vanila_D_loss = loss_discr_true + loss_discr_fake

    D_loss = vanila_D_loss + 1000 * gradient_penalty(discriminator, pseduo_output_, target_vflnn_pas_intermediate_, device)

    D_loss.backward()
    discriminator_optimizer.step()

    


    for para in target_vflnn.client1.parameters():
        para.requires_grad = True
    for para in target_vflnn.client2.parameters():
        para.requires_grad = True
    for para in target_vflnn.server.parameters():
        para.requires_grad = True
    for para in pseudo_model.parameters():
        para.requires_grad = True



    # 攻击测试，输出攻击图片和真实的mse
    if n % args.print_freq == 0:
        with torch.no_grad():
            attack_input = torch.cat((target_vflnn_pas_intermediate, target_vflnn_act_intermediate), cat_dimension)
            pseudo_attack_result = pseudo_inverse_model(attack_input)
        pseudo_target_mesloss = F.mse_loss(pseudo_attack_result, target_data)
        # print('Iter: %d / %d, Pseudo Loss: %.4f, Pseudo Inverse Loss: %.4f, Discriminator Loss: %.4f, Pseudo Target MSELoss: %.4f,  Dis_Pseudo_Loss: %.4f, Dis_target_Loss.: %.4f' % (n, 10000, pseudo_loss.item(), pseudo_inverse_loss.item(), D_loss.item(), pseudo_target_mesloss.item(), loss_discr_fake.item(), loss_discr_true.item()))

        logging.critical('Iter: %d / %d, Pseudo Loss: %.4f, Pseudo Inverse Loss: %.4f, Discriminator Loss: %.4f, Pseudo Target MSELoss: %.4f,  Dis_Pseudo_Loss: %.4f, Dis_target_Loss.: %.4f \n' % (n, 10000, pseudo_loss.item(), pseudo_inverse_loss.item(), D_loss.item(), pseudo_target_mesloss.item(), loss_discr_fake.item(), loss_discr_true.item()))


    return target_vflnn_pas_intermediate, target_vflnn_act_intermediate



# 攻击测试保存图片
def attack_test(pseduo_invmodel, target_data, target_vflnn_pas_intermediate, target_vflnn_act_intermediate, device, n):
    denorm = DeNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    with torch.no_grad():
        target_output = torch.cat((target_vflnn_pas_intermediate, target_vflnn_act_intermediate), 3)
        target_data = target_data.to(device)
        pseudo_inverse_result = pseduo_invmodel(target_output)
        # 去归一化
        # pseudo_inverse_result = denorm(pseudo_inverse_result.detach().clone())
        # origin_data = denorm(target_data.clone())
        truth = target_data[0:32]
        inverse_pseudo = pseudo_inverse_result[0:32]
        out_pseudo = torch.cat((inverse_pseudo, truth))

        for i in range(4):
            out_pseudo[i * 16:i * 16 + 8] = inverse_pseudo[i * 8:i * 8 + 8]
            out_pseudo[i * 16 + 8:i * 16 + 16] = truth[i * 8:i * 8 + 8]
        out_pseudo = denorm(out_pseudo.detach())
        pic_save_path = 'recon_pics2'
        os.makedirs('{}/pseudo'.format(pic_save_path),exist_ok=True)
        vutils.save_image(out_pseudo, '{}/pseudo/recon_{}.png'.format(pic_save_path,n), normalize=False)

# 测试整个vfl模型的准确率
def cal_test(target_vflnn, pseudo_model, test_loader, device, dataset):
    target_vflnn.eval()
    if pseudo_model is not None:
        pseudo_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            x_a, x_b = split_data(data, dataset)
            if pseudo_model is None:
                output = target_vflnn(x_a, x_b)
            elif pseudo_model is not None:
                pas_intermediate = pseudo_model(x_a)
                act_intermediate = target_vflnn.client2(x_b)
                output = target_vflnn.server([pas_intermediate, act_intermediate])
            # test loss
            test_loss += F.cross_entropy(output, target.long(), reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    # logging.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    return test_loss, 100. * correct / len(test_loader.dataset)




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