from utils import split_data, gradient_penalty, DeNormalize
import torch
from torch.nn import functional as F
import logging
from piq import ssim, psnr

def AGN_training(target_vflnn, pseudo_inverse_model, pseudo_inverse_optimizer, discriminator, discriminator_optimizer, target_data, target_label, shadow_data, device, n, cat_dimension, args):
    torch.autograd.set_detect_anomaly(True) # 检测梯度异常

    target_vflnn.client1.train()
    discriminator.train()
    pseudo_inverse_model.train()
    
    target_data = target_data.to(device)
    target_label = target_label.to(device)
    # 辅助数据shadow_data
    shadow_data = shadow_data.to(device)

    for para in target_vflnn.client2.parameters():
        para.requires_grad = False

    target_vflnn.client1_optimizer.zero_grad() 
    pseudo_inverse_optimizer.zero_grad()

    target_x_a, target_x_b = split_data(target_data, args.dataset)                       
    # 两个中间特征
    pas_intermediate = target_vflnn.client1(target_x_a) # 参与梯度计算
    act_intermediate = target_vflnn.client2(target_x_b).detach()
    # 两个特征合并
    pseudo_inverse_input = torch.cat((pas_intermediate, act_intermediate), cat_dimension)
    # 生成器(逆网络)生成的图片
    pseudo_inverse_output = pseudo_inverse_model(pseudo_inverse_input)
    # print(pseudo_inverse_output.shape) # [64,3, 32 ,32]
    
    # 被动方的损失，篡改，将生成的图片识别为1
    adv_pub_logits = discriminator(pseudo_inverse_output)
    client1_loss = torch.mean(adv_pub_logits)
    # 更新被动方模型参数
    client1_loss.backward()
    target_vflnn.client1_optimizer.step()
    # 更新生成器网络参数
    pseudo_inverse_optimizer.step()


    discriminator_optimizer.zero_grad()
    # 鉴别器的输出
    adv_priv_logits = discriminator(shadow_data) # 1
    adv_pub_logits = discriminator(pseudo_inverse_output.detach())  # 0
    # 鉴别器损失
    loss_discr_true = torch.mean(adv_priv_logits)
    loss_discr_fake = -torch.mean(adv_pub_logits)
    D_loss = loss_discr_true + loss_discr_fake + args.gan_p * gradient_penalty(discriminator, shadow_data, pseudo_inverse_output.detach(), device)
    # 更新鉴别器d的参数
    D_loss.backward()
    discriminator_optimizer.step()

    # 攻击测试
    if n % args.print_freq == 0:
        with torch.no_grad():
            attack_input = torch.cat((pas_intermediate, act_intermediate), cat_dimension)
            pseudo_attack_result = pseudo_inverse_model(attack_input)
        pseudo_target_mesloss = F.mse_loss(pseudo_attack_result, target_data)
        logging.critical('Iter: %d / %d, Discriminator Loss: %.4f, Pseudo Target MSELoss: %.4f,  Dis_Pseudo_Loss: %.4f, Dis_target_Loss.: %.4f' % (n, 10000, D_loss.clone().item(), pseudo_target_mesloss.item(), loss_discr_fake.item(), loss_discr_true.item()))

# 测试AGN的性能
def AGN_test(target_vflnn, pseudo_inverse_model, target_loader, device, args):
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
