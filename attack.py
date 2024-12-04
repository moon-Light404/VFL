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
from piq import ssim,psnr,LPIPS
import logging
from utils import split_data, gradient_penalty, DeNormalize

# 只训练鉴别器
def pseudo_training_1(target_vflnn, pseudo_model, pseudo_inverse_model, pseudo_optimizer, pseudo_inverse_optimizer, discriminator, discriminator_optimizer, target_data, target_label, shadow_data, shadow_label, device, n, cat_dimension, args):
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

    # 训练逆网络，同时优化伪模型和fa
    pseudo_inverse_optimizer.zero_grad()
    target_vflnn.client2_optimizer.zero_grad()
    with torch.no_grad():
        # pseudo_model 是client1的伪模型，伪模型和fa都更新
        pseudo_inverse_input_a = pseudo_model(shadow_x_a).detach()
        pseduo_inverse_input_b = target_vflnn.client2(shadow_x_b)
    # 两个特征拼接输入到逆网络，恢复数据 
    pseudo_inverse_input = torch.cat((pseudo_inverse_input_a, pseduo_inverse_input_b), cat_dimension)
    pseudo_inverse_output = pseudo_inverse_model(pseudo_inverse_input)
    pseudo_inverse_loss = F.mse_loss(pseudo_inverse_output, shadow_data)
    # 更新逆网络,fa
    pseudo_inverse_loss.backward()
    pseudo_inverse_optimizer.step()
    # 更新恶意方的底部模型
    target_vflnn.client2_optimizer.step()

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
    D_loss = vanila_D_loss + args.gan_p * gradient_penalty(discriminator, pseduo_output_, target_vflnn_pas_intermediate_, device)
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
        logging.critical('Iter: %d / %d, Pseudo Loss: %.4f, Pseudo Inverse Loss: %.4f, Discriminator Loss: %.4f, Pseudo Target MSELoss: %.4f,  Dis_Pseudo_Loss: %.4f, Dis_target_Loss.: %.4f \n' % (n, args.iteration, pseudo_loss.item(), pseudo_inverse_loss.item(), D_loss.item(), pseudo_target_mesloss.item(), loss_discr_fake.item(), loss_discr_true.item()))
    return target_vflnn_pas_intermediate, target_vflnn_act_intermediate

# 鉴别器+coral
def pseudo_training_2(target_vflnn, pseudo_model, pseudo_inverse_model, pseudo_optimizer, pseudo_inverse_optimizer, discriminator, discriminator_optimizer, target_data, target_label, shadow_data, shadow_label, device, n, cat_dimension, coral_loss, args):
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
    pseudo_output = pseudo_model(shadow_x_a) # 伪模型的特征空间
    # coral_loss计算
    n_domins = args.n_domins
    indices = [range(i, i + 64 // n_domins) for i in range(0, 64, 64 // n_domins)]
    coral_loss.train()
    loss_penalty = 0.0
    target = target_vflnn_pas_intermediate
    source = pseudo_output
    # coral_loss_item = coral_loss(target, source)
    
    if args.dataset == 'cifar10' or args.dataset == 'tinyImagenet': 
        for domin_i in range(n_domins):
            for domin_j in range(n_domins):
                for i in range(64 // n_domins):
                    f_i = target[indices[domin_i][i], :, :, :].view(target.size(1),-1)
                    f_j = pseudo_output[indices[domin_j][i], :, :, :].view(target.size(1),-1)
                    loss_penalty += coral_loss(f_i,f_j)
        loss_penalty /= n_domins * n_domins * (64 // n_domins)
      
    else:
        target = target.chunk(n_domins, dim=0)
        source = pseudo_output.chunk(n_domins, dim=0)

        for domin_i in range(n_domins):
            for domin_j in range(n_domins):
                f_i = target[domin_i]
                f_j = source[domin_j]
                loss_penalty += coral_loss(f_i,f_j)
        loss_penalty /= n_domins * n_domins
    
    if n % args.print_freq == 0:
        logging.critical('Coral Loss: %.4f' % (loss_penalty.item()))
     # 把伪模型的特征空间输入到鉴别器中
    d_output_pseudo = discriminator(pseudo_output)
    pseudo_loss = (1 - args.a) * torch.mean(d_output_pseudo) + args.a * loss_penalty
    pseudo_loss.backward()
    pseudo_optimizer.step()

    # 训练逆网络
    pseudo_inverse_optimizer.zero_grad()
    with torch.no_grad():
        # pseudo_model 是client1的伪模型，伪模型和fa都更新
        pseudo_inverse_input_a = pseudo_model(shadow_x_a).detach()
        pseduo_inverse_input_b = target_vflnn.client2(shadow_x_b)
    # 两个特征拼接输入到逆网络，恢复数据 
    pseudo_inverse_input = torch.cat((pseudo_inverse_input_a, pseduo_inverse_input_b), cat_dimension)
    pseudo_inverse_output = pseudo_inverse_model(pseudo_inverse_input)
    pseudo_inverse_loss = F.mse_loss(pseudo_inverse_output, shadow_data)
    # 更新逆网络,fa
    pseudo_inverse_loss.backward()
    pseudo_inverse_optimizer.step()

    # 更新鉴别器，此时不能更新伪模型，设为detach()
    discriminator_optimizer.zero_grad()
    pseduo_output_ = pseudo_output.detach()
    # 目标客户端特征空间
    target_vflnn_pas_intermediate_ = target_vflnn_pas_intermediate.detach()
    adv_target_logits = discriminator(target_vflnn_pas_intermediate_)
    adv_pseudo_logits = discriminator(pseduo_output_)
    loss_discr_true = torch.mean(adv_target_logits)
    loss_discr_fake = -torch.mean(adv_pseudo_logits)
    vanila_D_loss = loss_discr_true + loss_discr_fake
    D_loss = vanila_D_loss + args.gan_p * gradient_penalty(discriminator, pseduo_output_, target_vflnn_pas_intermediate_, device)
    D_loss.backward()
    discriminator_optimizer.step()

    for para in target_vflnn.client1.parameters():
        para.requires_grad = True
    for para in target_vflnn.client2.parameters():
        para.requires_grad = True
    for para in target_vflnn.server.parameters():
        para.requires_grad = True
    # 攻击测试，输出攻击图片和真实的mse
    if n % args.print_freq == 0:
        with torch.no_grad():
            attack_input = torch.cat((target_vflnn_pas_intermediate, target_vflnn_act_intermediate), cat_dimension)
            pseudo_attack_result = pseudo_inverse_model(attack_input)
        pseudo_target_mesloss = F.mse_loss(pseudo_attack_result, target_data)
        logging.critical('Iter: %d / %d, Pseudo Loss: %.4f, Pseudo Inverse Loss: %.4f, Discriminator Loss: %.4f, Pseudo Target MSELoss: %.4f,  Dis_Pseudo_Loss: %.4f, Dis_target_Loss.: %.4f \n' % (n, args.iteration, pseudo_loss.item(), pseudo_inverse_loss.item(), D_loss.item(), pseudo_target_mesloss.item(), loss_discr_fake.item(), loss_discr_true.item()))
    return target_vflnn_pas_intermediate, target_vflnn_act_intermediate

# 鉴别器+pcat
def pseudo_training_3(target_vflnn, pseudo_model, pseudo_inverse_model, pseudo_optimizer, pseudo_inverse_optimizer, discriminator, discriminator_optimizer, target_data, target_label, shadow_data, shadow_label, device, n, cat_dimension, args):
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

    # 训练逆网络，同时优化伪模型和fa
    pseudo_inverse_optimizer.zero_grad()
    target_vflnn.client2_optimizer.zero_grad()
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

    # 进一步更新伪模型，只更新伪模型
    for para in target_vflnn.client2.parameters():
        para.requires_grad = False
    # 学习服务器的知识
    # loss_flag = 1000
    # with torch.no_grad():
    #     vflnn_output = target_vflnn(target_x_a, target_x_b)
    #     target_vflnn_loss = F.cross_entropy(vflnn_output, target_label)
    #     loss_flag = target_vflnn_loss.clone().item()
    # loss_flag < 0 就不进入这个分支
    # if args.loss_threshold > 0 and loss_flag < args.loss_threshold:
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
    D_loss = vanila_D_loss + args.gan_p * gradient_penalty(discriminator, pseduo_output_, target_vflnn_pas_intermediate_, device)
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
        logging.critical('Iter: %d / %d, Pseudo Loss: %.4f, Pseudo Inverse Loss: %.4f, Discriminator Loss: %.4f, Pseudo Target MSELoss: %.4f,  Dis_Pseudo_Loss: %.4f, Dis_target_Loss.: %.4f \n' % (n, args.iteration, pseudo_loss.item(), pseudo_inverse_loss.item(), D_loss.item(), pseudo_target_mesloss.item(), loss_discr_fake.item(), loss_discr_true.item()))
    return target_vflnn_pas_intermediate, target_vflnn_act_intermediate

# 只有pcat
def pseudo_training_4(target_vflnn, pseudo_model, pseudo_inverse_model, pseudo_optimizer, pseudo_inverse_optimizer, target_data, target_label, shadow_data, shadow_label, device, n, cat_dimension, args):
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
    for para in target_vflnn.client2.parameters():
        para.requires_grad = False
    for para in target_vflnn.server.parameters():
        para.requires_grad = False  
    
    # 训练伪模型，比较的特征空间是图像的一半
    pseudo_inverse_model.train()
    shadow_data = shadow_data.to(device)
    shadow_label = shadow_label.to(device)
    # 辅助数据分成两半，一般给伪模型，一般给主动方的底部模型fa
    shadow_x_a, shadow_x_b = split_data(shadow_data, args.dataset)
    
    # 训练逆网络，同时优化fa
    pseudo_inverse_optimizer.zero_grad()
    with torch.no_grad():
        # pseudo_model 是client1的伪模型，fa都更新
        pseudo_inverse_input_a = pseudo_model(shadow_x_a)
        pseduo_inverse_input_b = target_vflnn.client2(shadow_x_b)
    # 两个特征拼接输入到逆网络，恢复数据 
    pseudo_inverse_input = torch.cat((pseudo_inverse_input_a, pseduo_inverse_input_b), cat_dimension)
    pseudo_inverse_output = pseudo_inverse_model(pseudo_inverse_input)
    pseudo_inverse_loss = F.mse_loss(pseudo_inverse_output, shadow_data)
    # 更新逆网络、fa
    pseudo_inverse_loss.backward()
    pseudo_inverse_optimizer.step()
    # 更新恶意方的底部模型

    pseudo_model.train()
    # 进一步更新伪模型，只更新伪模型
    # 学习服务器的知识
    # loss_flag = 1000
    # with torch.no_grad():
    #     vflnn_output = target_vflnn(target_x_a, target_x_b)
    #     target_vflnn_loss = F.cross_entropy(vflnn_output, target_label)
    #     loss_flag = target_vflnn_loss.clone().item()
    # loss_flag < 0 就不进入这个分支
    # if args.loss_threshold > 0 and loss_flag < args.loss_threshold:
    target_vflnn.zero_grads()
    pseudo_optimizer.zero_grad()
    target_vflnn.eval()
    classifier_input1 = pseudo_model(shadow_x_a)
    classifier_input2 = target_vflnn.client2(shadow_x_b).detach()
    classifer_output = target_vflnn.server([classifier_input1, classifier_input2])
    classifier_loss = F.cross_entropy(classifer_output, shadow_label)
    classifier_loss.backward()
    pseudo_optimizer.step() # 更新伪模型,更接近被动方模型

    for para in target_vflnn.client1.parameters():
        para.requires_grad = True
    for para in target_vflnn.client2.parameters():
        para.requires_grad = True
    for para in target_vflnn.server.parameters():
        para.requires_grad = True

    # 攻击测试，输出攻击图片和真实的mse
    if n % args.print_freq == 0:
        with torch.no_grad():
            attack_input = torch.cat((target_vflnn_pas_intermediate, target_vflnn_act_intermediate), cat_dimension)
            pseudo_attack_result = pseudo_inverse_model(attack_input)
        pseudo_target_mesloss = F.mse_loss(pseudo_attack_result, target_data)
        logging.critical('Iter: %d / %d,  Pseudo Inverse Loss: %.4f, Pseudo Target MSELoss: %.4f' % (n, args.iteration, pseudo_inverse_loss.item(), pseudo_target_mesloss.item()))
    return target_vflnn_pas_intermediate, target_vflnn_act_intermediate

# 鉴别器+coral+pcat
def pseudo_training_5(target_vflnn, pseudo_model, pseudo_inverse_model, pseudo_optimizer, pseudo_inverse_optimizer, discriminator, discriminator_optimizer, target_data, target_label, shadow_data, shadow_label, device, n, cat_dimension, coral_loss, args):
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
    pseudo_output = pseudo_model(shadow_x_a) # 伪模型的特征空间
    n_domins = args.n_domins
    indices = [range(i, i + 64 // n_domins) for i in range(0, 64, 64 // n_domins)]
    coral_loss.train()
    loss_penalty = 0.0
    target = target_vflnn_pas_intermediate
    source = pseudo_output
    if args.dataset == 'cifar10' or args.dataset == 'tinyImagenet':
        for domin_i in range(n_domins):
            for domin_j in range(n_domins):
                for i in range(64 // n_domins):
                    f_i = target[indices[domin_i][i], :, :, :].view(target.size(1),-1)
                    f_j = pseudo_output[indices[domin_j][i], :, :, :].view(target.size(1),-1)
                    loss_penalty += coral_loss(f_i,f_j)
        loss_penalty /= n_domins * n_domins * (64 // n_domins)
    else:
        target = target.chunk(n_domins, dim=0)
        source = pseudo_output.chunk(n_domins, dim=0)

        for domin_i in range(n_domins):
            for domin_j in range(n_domins):
                f_i = target[domin_i]
                f_j = source[domin_j]
                loss_penalty += coral_loss(f_i,f_j)
        loss_penalty /= n_domins * n_domins * (64 // n_domins)
    if n % args.print_freq == 0:
        logging.critical('Coral Loss: %.4f' % (loss_penalty.item()))
    # 把伪模型的特征空间输入到鉴别器中
    d_output_pseudo = discriminator(pseudo_output)
    pseudo_loss = (1 - args.a) * torch.mean(d_output_pseudo) + args.a * loss_penalty

    pseudo_loss.backward()
    pseudo_optimizer.step()

    # 训练逆网络，同时优化伪模型和fa
    pseudo_inverse_optimizer.zero_grad()
    target_vflnn.client2_optimizer.zero_grad()
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

    # 进一步更新伪模型，只更新伪模型
    for para in target_vflnn.client2.parameters():
        para.requires_grad = False
    
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
    
    pseduo_output_ = pseudo_output.detach()
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
        logging.critical('Iter: %d / %d, Pseudo Loss: %.4f, Pseudo Inverse Loss: %.4f, Discriminator Loss: %.4f, Pseudo Target MSELoss: %.4f,  Dis_Pseudo_Loss: %.4f, Dis_target_Loss.: %.4f \n' % (n, args.iteration, pseudo_loss.item(), pseudo_inverse_loss.item(), D_loss.item(), pseudo_target_mesloss.item(), loss_discr_fake.item(), loss_discr_true.item()))
    return target_vflnn_pas_intermediate, target_vflnn_act_intermediate

    

# 攻击测试保存图片
def attack_test(pseduo_invmodel, pseudo_model, target_vflnn, target_data, target_vflnn_pas_intermediate, target_vflnn_act_intermediate, device, args, n):
    if args.dataset == 'tinyImagenet':
        denorm = DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif args.dataset == 'cifar10':
        denorm = DeNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    target_pseudo_loss = 0
    pseudo_ssim = 0
    pseudo_psnr = 0
    pseudo_llips = 0

    with torch.no_grad():
        target_output = torch.cat((target_vflnn_pas_intermediate, target_vflnn_act_intermediate), 3)
        target_data = target_data.to(device)
        pseudo_inverse_output = pseduo_invmodel(target_output)

        origin_data = denorm(target_data.clone())
        pseudo_inverse_output_ = denorm(pseudo_inverse_output.clone())
        pseudo_ssim += ssim(origin_data, pseudo_inverse_output_, reduction='sum').item()
        pseudo_psnr += psnr(origin_data, pseudo_inverse_output_, reduction='sum').item()
        # pseudo_llips += LPIPS(origin_data, pseudo_inverse_output_, reduction='sum').item()

        x_a, x_b = split_data(target_data, args.dataset)
        pseudo_intermediate = pseudo_model(x_a)
        target_intermediate = target_vflnn.client1(x_a)
        target_pseudo_loss += F.mse_loss(pseudo_intermediate, target_intermediate, reduce='sum').item()

        # 保存图片
        # if n % 100 == 0:
        # truth = target_data[0:32]
        # inverse_pseudo = pseudo_inverse_result[0:32]
        # out_pseudo = torch.cat((inverse_pseudo, truth))

        # for i in range(4):
        #     out_pseudo[i * 16:i * 16 + 8] = inverse_pseudo[i * 8:i * 8 + 8]
        #     out_pseudo[i * 16 + 8:i * 16 + 16] = truth[i * 8:i * 8 + 8]
        # out_pseudo = denorm(out_pseudo.detach())
        # pic_save_path = 'recon_pics2'
        # os.makedirs('{}/pseudo'.format(pic_save_path),exist_ok=True)
        # vutils.save_image(out_pseudo, '{}/pseudo/recon_{}.png'.format(pic_save_path,n), normalize=False)

    pseudo_ssim /= len(target_data)
    pseudo_psnr /= len(target_data)
    # pseudo_llips /= len(target_data)
    target_pseudo_loss /= len(target_data)
    return target_pseudo_loss, pseudo_ssim, pseudo_psnr

# 训练完成后测试逆网络恢复的性能mse psnr ssim
def attack_test_all(target_vflnn, pseudo_inverse_model, target_loader, device, dataset):
    if dataset == 'tinyImagenet':
        denorm = DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif dataset == 'cifar10':
        denorm = DeNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    target_vflnn.eval()
    pseudo_inverse_model.eval()
    mse_loss = []
    psnr_loss = []
    ssim_loss = []
    with torch.no_grad():
        for data, target in target_loader:
            data, target = data.to(device), target.to(device)
            x_a, x_b = split_data(data, dataset)
            target_output = target_vflnn(x_a, x_b)
            pseudo_inverse_input = target_vflnn.server.input
            recover_data = pseudo_inverse_model(pseudo_inverse_input)
            recover_loss = F.mse_loss(recover_data, data, reduction='mean').item()
            mse_loss.append(recover_loss)
            origin_data_ = denorm(data.clone())
            recover_data_ = denorm(recover_data.clone())
            ssim_ = ssim(origin_data_, recover_data_, reduction='mean').item()
            psnr_ = psnr(origin_data_, recover_data_, reduction='mean').item()
            psnr_loss.append(psnr_)
            ssim_loss.append(ssim_)
    mean_loss = sum(mse_loss) / len(mse_loss)
    mean_psnr = sum(psnr_loss) / len(psnr_loss)
    mean_ssim = sum(ssim_loss) / len(ssim_loss)
    return mean_loss, mean_psnr, mean_ssim


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
    return test_loss, 100. * correct / len(test_loader.dataset)

# 特征之间的余弦相似度
def cosine_similarity(pseudo_model, client, target_data, device, dataset):
    pseudo_model.eval()
    client.eval()
    with torch.no_grad():
        target_data = target_data.to(device)
        x_a, _ = split_data(target_data, dataset)
        pseudo_output = pseudo_model(x_a)
        client_output = client(x_a)
    # 将特征展平
    pseudo_output = pseudo_output.view(pseudo_output.size(0), -1)
    client_output = client_output.view(client_output.size(0), -1)
    similarity = F.cosine_similarity(pseudo_output, client_output, dim=1)
    return similarity.mean().item()

# 特征之间的均方误差
def mean_squared_error_loss(pseudo_model, client, target_data, device, dataset):
    pseudo_model.eval()
    client.eval()
    with torch.no_grad():
        target_data = target_data.to(device)
        x_a, _ = split_data(target_data, dataset)
        pseudo_output = pseudo_model(x_a)
        client_output = client(x_a)
    # 将特征展平
    pseudo_output = pseudo_output.view(pseudo_output.size(0), -1)
    client_output = client_output.view(client_output.size(0), -1)
    # 计算均方误差损失
    mse_loss = F.mse_loss(pseudo_output, client_output, reduction='mean')
    return mse_loss.item()



    

