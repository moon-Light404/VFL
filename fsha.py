from utils import split_data, gradient_penalty, DeNormalize
from torch.nn import functional as F
import torch
import torch.nn as nn
import logging
from piq import ssim, psnr

def fsha(pas_client, act_client, pseudo_model, decoder, discriminator, pas_client_optimizer, pseudo_model_optimizer, decoder_optimizer, discriminator_optimizer, target_data, target_label, device, shadow_data, shadow_label, n, cat_dimension, args):

    torch.autograd.set_detect_anomaly(True)

    target_data, target_label = target_data.to(device), target_label.to(device)

    pas_client.train()
    pseudo_model.train()
    decoder.train()

    pas_client_optimizer.zero_grad()
    target_data_a, target_data_b = split_data(target_data, args.dataset)

    act_intermediate = act_client(target_data_b).detach()
    pas_intermediate = pas_client(target_data_a)

    adv_target_logits = discriminator(pas_intermediate)
    # 更新被动方的模型参数(恶意梯度)
    pas_client_loss = torch.mean(adv_target_logits)
    pas_client_loss.backward()
    pas_client_optimizer.step()

    
    pseudo_model_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    shadow_data, shadow_label = shadow_data.to(device), shadow_label.to(device)
    shadow_data_a, shadow_data_b = split_data(shadow_data, args.dataset)
    # 模仿被动方的伪模型
    pseudo_output_a = pseudo_model(shadow_data_a)
    pseudo_output_b = act_client(shadow_data_b).detach()
    decoder_input = torch.cat((pseudo_output_a, pseudo_output_b), cat_dimension)
    # 解码器的输出，恢复的被动方输出
    rec_shadow_data = decoder(decoder_input)
    rec_loss = F.mse_loss(rec_shadow_data, shadow_data)
    # 更新伪模型和解码器的参数
    rec_loss.backward()
    # 同时更新伪模型和解码器
    pseudo_model_optimizer.step()
    decoder_optimizer.step()

    discriminator.train()
    discriminator_optimizer.zero_grad()

    adv_target_logits = discriminator(pas_intermediate.detach())
    adv_pseudo_logits = discriminator(pseudo_output_a.detach())

    loss_discr_true = torch.mean(adv_pseudo_logits)
    loss_discr_fake = -torch.mean(adv_target_logits)  
    van_D_loss = loss_discr_true + loss_discr_fake
    # 更新鉴别器的参数

    D_loss = van_D_loss + args.gan_p * gradient_penalty(discriminator, pas_intermediate.detach(), pseudo_output_a.detach(), device)
    D_loss.backward()
    discriminator_optimizer.step()  

    if n % args.print_freq == 0:
        with torch.no_grad():
            attack_input = torch.cat((pas_intermediate, act_intermediate), cat_dimension)
            pseudo_attack_result = decoder(attack_input)
        pseudo_target_mesloss = F.mse_loss(pseudo_attack_result, target_data)
        logging.critical('Iter: %d / %d,  Pseudo Inverse Loss: %.4f, Discriminator Loss: %.4f, Pseudo Target MSELoss: %.4f,  Dis_Pseudo_Loss: %.4f, Dis_target_Loss.: %.4f' % (n, args.iteration,  rec_loss.item(), D_loss.item(), pseudo_target_mesloss.item(), loss_discr_fake.item(), loss_discr_true.item()))

def FSHA_test(target_vflnn, pseudo_inverse_model, target_loader, device, args):
    denorm = DeNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    target_vflnn.eval()
    pseudo_inverse_model.eval()
    mse_loss = []
    psnr_loss = []
    ssim_loss = []
    with torch.no_grad():
        for data, target in target_loader:
            data, target = data.to(device), target.to(device)
            x_a, x_b = split_data(data, args.dataset)
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





