import argparse
import os
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import copy
import torchvision.utils as vutils
import torchvision.models as models
from vfl import Client, Server, VFLNN
from attack import attack_test, pseudo_training_1, pseudo_training_2, pseudo_training_3, pseudo_training_4, pseudo_training_5, cal_test, attack_test_all
from model import cifar_mobilenet, cifar_decoder, cifar_discriminator_model, vgg16, cifar_pseudo, bank_net, bank_pseudo, bank_discriminator,bank_decoder,resnet_from_model, resnet_decoder, resnet_discriminator, Resnet
import numpy as np
from torch.utils.data import Subset
from random import shuffle
import math
from agn import AGN_training
from fsha import fsha
from datasets import ExperimentDataset, getSplittedDataset
import time
import logging
import argparse
import pytz
from datetime import datetime
from logging import Formatter
from utils import CorrelationAlignmentLoss

# 设置时区为北京时间
class BeijingFormatter(Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, pytz.timezone('Asia/Shanghai'))
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            s = dt.isoformat()
        return s

# INFO级别以上的日志会记录到日志文件，critical级别的日志会输出到控制台
def initlogging(logfile):
    # debug, info, warning, error, critical
    # set up logging to file
    logging.shutdown()
    
    logger = logging.getLogger()
    logger.handlers = []
    # 设置日志记录级别为INFO，即只有INFO级别及以上的会被记录
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=logfile,
                        filemode='w')
    
    for handler in logging.getLogger().handlers:
        handler.setFormatter(BeijingFormatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.CRITICAL) # 只有critical级别的才会输出到控制台
    # add formatter to ch
    ch.setFormatter(logging.Formatter('%(message)s')) # 控制台只输出日志消息的内容
    logging.getLogger().addHandler(ch)  

def save_model(model, path):
    os.makedirs(os.path.join(path, 'pseudo'), exist_ok=True)
    os.makedirs(os.path.join(path, 'pseudo_inverse_model'), exist_ok=True)
    

def main():
    parser = argparse.ArgumentParser(description="VFL of implementation")
    parser.add_argument('--iteration', type=int, default=10000, help="")
    parser.add_argument('--lr', type=float, default=1e-4, help="the learning rate of pseudo_inverse model")
    parser.add_argument('--dlr', type=float, default=3e-4, help="the learning rate of discriminate")
    parser.add_argument('--batch_size', type=int, default=64, help="")
    parser.add_argument('--print_freq', type=int, default='25', help="the print frequency of ouput")
    parser.add_argument('--dataset', type=str, default='cifar10', help="the test dataset bank cifar10 tinyImagenet")
    parser.add_argument('--level', type=int, default=2, help="the split layer of model")
    parser.add_argument('--dataset_portion', type=float, default=0.05, help="the size portion of auxiliary data")
    parser.add_argument('--train_portion', type=float, default=0.7, help="the train_data portion of bank/drive data")
    parser.add_argument('--test_portion', type=float, default=0.3, help="the test portion of bank.drive data")
    parser.add_argument('--attack', type=str, default='our', help="the type of attack agn, our, fsha, grna")
    parser.add_argument('--loss_threshold', type=float, default=1.7, help="the loss flag of our attack")
    parser.add_argument('--n_domins', type=int, default=4, help="the domins of save each epoch")
    # 1-鉴别器 2-鉴别器+coral 3-鉴别器+pcat 4-pcat 5-鉴别器+coral+pcat
    parser.add_argument('--pseudo_train', type=int, choices=[1, 2, 3, 4, 5], default=2, help="the type of training")
    parser.add_argument('--a', type=float, default=0.3, help="the weight of coral")
    parser.add_argument('--gan_p', type=float, default=1000.0, help="the weight of wgan")

    args = parser.parse_args()

    gid = '0'
    date_time_file = datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d-%H-%M")


    if args.dataset == 'bank':
        dataset_path = 'data/bank_cleaned.csv'
        vfl_input_dim = 10
        vfl_output_dim = 2
        cat_dimension = 1 # 拼接维度
    elif args.dataset == 'drive':
        dataset_path = 'data/drive_cleaned.csv'
        vfl_input_dim = 24
        vfl_output_dim = 11
        cat_dimension = 1 # 拼接维度
    # dataset_num = 1524
    
     # 固定初始化，可重复性
    torch.manual_seed(3407)
    random.seed(3407)
    np.random.seed(3407)
    cudnn.deterministic = True
    cudnn.benchmark = False

    level = 'level' + str(args.level)
    n_domins = 'n_domins' + str(args.n_domins)
    pseudo_train = 'pseudo_train' + str(args.pseudo_train)
    
    path_name = os.path.join('log', args.attack, args.dataset, level)
    if args.attack == 'our':
        path_name = os.path.join(path_name, pseudo_train)
        if args.pseudo_train == 2:
            path_name = os.path.join(path_name, n_domins)
    os.makedirs(path_name, exist_ok=True)
    initlogging(logfile=os.path.join(path_name, date_time_file + '.log'))
    logging.info(">>>>>>>>>>>>>>Running settings>>>>>>>>>>>>>>")
    for arg in vars(args):
        logging.info("%s: %s", arg, getattr(args, arg))
    logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n")

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        id = 'cuda:'+ gid
        device = torch.device(id)
        torch.cuda.set_device(id)
        # cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    print(device)
   

    cifar_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            ])
    tiny_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    if args.dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train = True, transform=cifar_transform, download=False)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train = False, transform=cifar_transform, download=False)
        # 取2500/5000个私有数据
        dataset_num = len(train_dataset) * args.dataset_portion
        shadow_dataset = Subset(test_dataset, range(0, int(dataset_num)))
        cat_dimension = 3
    elif args.dataset == 'tinyImagenet':
        train_dataset = torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/train', 
                                                         transform=transforms.Compose([transforms.ToTensor(),
                                                          tiny_normalize])
                                                         )
        test_dataset = torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/val', 
                                                        transform= transforms.Compose([transforms.ToTensor(),
                                                            tiny_normalize])
                                                        )
        shadow_dataset = torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/test', 
                                                        transform= transforms.Compose([transforms.ToTensor(),
                                                            tiny_normalize])
                                                        )
        
        dataset_num = len(train_dataset) * args.dataset_portion
        shadow_dataset = Subset(shadow_dataset, range(0, int(dataset_num)))
        cat_dimension = 3
    else: # bank 数据集
        bank_expset = ExperimentDataset(datafilepath=dataset_path)
        train_dataset, test_dataset = getSplittedDataset(args.train_portion, args.test_portion, bank_expset)
        dataset_num = len(train_dataset) * args.dataset_portion
        shadow_dataset = Subset(test_dataset, range(0, int(dataset_num)))
        cat_dimension = 1
   

    logging.info("DataSet:%s", args.dataset)
    logging.info("Train Dataset: %d",len(train_dataset))
    logging.info("Test Dataset: %d",len(test_dataset))
    logging.info("Shadow Dataset:%d",len(shadow_dataset))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size= args.batch_size, shuffle=True, num_workers = 8, pin_memory = True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = 8, pin_memory = True)
    shadow_dataloader = torch.utils.data.DataLoader(shadow_dataset, batch_size=args.batch_size, shuffle=True, num_workers = 8, pin_memory = True)


    # 构建模型
    if args.dataset == 'cifar10':
        target_bottom1, target_top = cifar_mobilenet(args.level)
        target_bottom2 = copy.deepcopy(target_bottom1)
        data_shape = train_dataset[0][0].shape
        pseudo_model, _ = vgg16(args.level, batch_norm = True)
        test_data = torch.ones(1,data_shape[0], data_shape[1], data_shape[2])
        with torch.no_grad():
            test_data_output = pseudo_model(test_data)
            discriminator_input_shape = test_data_output.shape[1:] # 除去第0维以后的维度，0维是批次大小
        print(discriminator_input_shape) # 中间特征大小

        d_input_shape = 3 if args.attack == 'agn' else discriminator_input_shape[0]
        agn = True if args.attack == 'agn' else False
        # 初始化鉴别器, agn==3
        discriminator = cifar_discriminator_model(d_input_shape, args.level, agn)
        # 初始化逆网络(inchannel, levle, outchannel)
        pseudo_inverse_model = cifar_decoder(discriminator_input_shape, args.level, 3)
    elif args.dataset == 'tinyImagenet':
        model_ft = models.resnet18()
        model_path = 'resnet18-f37072fd.pth'
        model_ft.load_state_dict(torch.load(model_path))
        target_bottom1, target_bottom2, target_top = resnet_from_model(model_ft, args.level)
        data_shape = train_dataset[0][0].shape
        test_data = torch.ones(1,data_shape[0], data_shape[1], data_shape[2])
        pseudo_model = Resnet(args.level)
        with torch.no_grad():
            test_data_output = pseudo_model(test_data)
            discriminator_input_shape = test_data_output.shape[1:]
        print(discriminator_input_shape)
        d_input_shape = discriminator_input_shape[0]
        discriminator = resnet_discriminator(d_input_shape, args.level)
        pseudo_inverse_model = resnet_decoder(d_input_shape, args.level, 3)
    else:
        target_bottom1, target_top = bank_net(input_dim=vfl_input_dim, output_dim=vfl_output_dim)
        target_bottom2 = copy.deepcopy(target_bottom1)
        test_data = torch.ones(1, vfl_input_dim)
        with torch.no_grad():
            test_data_output = target_bottom1(test_data)
            d_input_shape = test_data_output.shape[1] # d_input_shape是被动客户端输入的中间特征
        pseudo_model = bank_pseudo(input_dim=vfl_input_dim, output_dim = d_input_shape)
        if args.attack == 'agn': # 鉴别器输入的是完整的数据记录
            d_input_shape = vfl_input_dim * 2
        discriminator = bank_discriminator(input_dim = d_input_shape)
        d_input_shape = test_data_output.shape[1]
        # decoder的输入是两个中间特征拼接
        pseudo_inverse_model = bank_decoder(input_dim = 2 * d_input_shape, output_dim = vfl_input_dim * 2)

    target_bottom1, target_bottom2, target_top = target_bottom1.to(device), target_bottom2.to(device), target_top.to(device)
    pseudo_model = pseudo_model.to(device)
    discriminator = discriminator.to(device)
    pseudo_inverse_model = pseudo_inverse_model.to(device)

    # 初始化服务器和客户端
    pas_client = Client(target_bottom1)
    act_client = Client(target_bottom2)
    act_server = Server(target_top, cat_dimension)

    # 初始化优化器
    # 对于cifar10 lr dlr都是1e-4
    # if args.dataset == 'cifar10':
    pas_client_optimizer = optim.Adam(target_bottom1.parameters(), lr=1e-3)
    act_client_optimizer = optim.Adam(target_bottom2.parameters(), lr=1e-3)
    act_server_optimizer = optim.Adam(target_top.parameters(), lr=1e-3)
    pseudo_optimizer = optim.Adam(pseudo_model.parameters(), lr=args.lr)
    pseudo_inverse_model_optimizer = optim.Adam(pseudo_inverse_model.parameters(), lr=args.lr)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=args.dlr)

    target_vflnn = VFLNN(pas_client, act_client, act_server, [pas_client_optimizer, act_client_optimizer], act_server_optimizer)
    target_iterator = iter(train_dataloader)
    shadow_iterator = iter(shadow_dataloader)

    coral_loss = CorrelationAlignmentLoss()



    for n in range(1, args.iteration+1):
        if (n-1)%int((len(train_dataset)/args.batch_size)) == 0 :        
            target_iterator = iter(train_dataloader) # 从头开始迭代
        if (n-1)%int((len(shadow_dataset)/args.batch_size)) == 0 :        
            shadow_iterator = iter(shadow_dataloader) # 从头开始迭代         
        try:
            target_data, target_label = next(target_iterator)
        except StopIteration:
            target_iterator = iter(train_dataloader)
            target_data, target_label = next(target_iterator)
        try:     
            shadow_data, shadow_label = next(shadow_iterator)
        except StopIteration:
            shadow_iterator = iter(shadow_dataloader)
            shadow_data, shadow_label = next(shadow_iterator)
        if target_data.size(0) != shadow_data.size(0):
            print("The number is not match")
            exit() 
        if args.dataset == 'bank' or args.dataset == 'drive':
            target_label = target_label.long()
            shadow_label = shadow_label.long()
        
        if args.attack == 'agn':
            # AGN攻击测试
            AGN_training(target_vflnn, pseudo_inverse_model, pseudo_inverse_model_optimizer, discriminator, discriminator_optimizer, target_data, target_label, shadow_data, device, n, cat_dimension, args)
        elif args.attack == 'fsha':
            fsha(pas_client, act_client, pseudo_model, pseudo_inverse_model, discriminator, pas_client_optimizer,pseudo_optimizer, pseudo_inverse_model_optimizer, discriminator_optimizer, target_data, target_label, device, shadow_data, shadow_label, n, cat_dimension, args)
        elif args.attack == 'our':
            if args.pseudo_train == 1:
                target_vflnn_pas_intermediate, target_vflnn_act_intermediate = pseudo_training_1(target_vflnn, pseudo_model, pseudo_inverse_model, pseudo_optimizer, pseudo_inverse_model_optimizer, discriminator, discriminator_optimizer, target_data, target_label, shadow_data, shadow_label, device, n, cat_dimension, args)
                
                
            elif args.pseudo_train == 2:
                target_vflnn_pas_intermediate, target_vflnn_act_intermediate = pseudo_training_2(target_vflnn, pseudo_model, pseudo_inverse_model, pseudo_optimizer, pseudo_inverse_model_optimizer, discriminator, discriminator_optimizer, target_data, target_label, shadow_data, shadow_label, device, n, cat_dimension, coral_loss, args)
                    
                    
            elif args.pseudo_train == 3:
                target_vflnn_pas_intermediate, target_vflnn_act_intermediate = pseudo_training_3(target_vflnn, pseudo_model, pseudo_inverse_model, pseudo_optimizer, pseudo_inverse_model_optimizer, discriminator, discriminator_optimizer, target_data, target_label, shadow_data, shadow_label, device, n, cat_dimension, args)
                
                
            elif args.pseudo_train == 4:
                target_vflnn_pas_intermediate, target_vflnn_act_intermediate = pseudo_training_4(target_vflnn, pseudo_model, pseudo_inverse_model, pseudo_optimizer, pseudo_inverse_model_optimizer, target_data, target_label, shadow_data, shadow_label, device, n, cat_dimension, args)
                
                
            elif args.pseudo_train == 5:
                target_vflnn_pas_intermediate, target_vflnn_act_intermediate = pseudo_training_5(target_vflnn, pseudo_model, pseudo_inverse_model, pseudo_optimizer, pseudo_inverse_model_optimizer, discriminator, discriminator_optimizer, target_data, target_label, shadow_data, shadow_label, device, n, cat_dimension, coral_loss, args)
                
                
            # 每隔100次迭代进行攻击测试，保存图片
            if (args.dataset == 'cifar10' or args.dataset == 'tinyImagenet') and n > 6000 and n % 10 == 0:
                target_pseudo_loss, pseudo_ssim, pseudo_psnr = attack_test(pseudo_inverse_model, pseudo_model, target_vflnn, target_data, target_vflnn_pas_intermediate, target_vflnn_act_intermediate, device, args, n)
                logging.critical("Iter: %d / %d, Pseudo SSIM: %.4f, Pseudo PSNR: %.4f" %(n, args.iteration,pseudo_ssim, pseudo_psnr))
                if pseudo_psnr < 15:
                    break
                
                
            # 下面测试伪模型的实用性
            if n % 50 == 0:
                # 正常VFL测试
                logging.critical("Start testing the accuracy of the model: \n")
                vfl_loss, vfl_acc = cal_test(target_vflnn, None, test_dataloader, device, args.dataset)
                # 伪被动客户端VFL测试
                pseudo_loss, pseudo_acc = cal_test(target_vflnn, pseudo_model, test_dataloader, device, args.dataset)
                logging.critical("VFL Loss: {:.4f}, VFL Acc: {:.4f},\n Pseudo Loss: {:.4f}, Pseudo Acc: {:.4f}".format(vfl_loss, vfl_acc, pseudo_loss, pseudo_acc))
                
            if n % 5000 == 0:
                mean_loss, mean_psnr, mean_ssim = attack_test_all(target_vflnn, pseudo_inverse_model, train_dataloader, device, args)
                logging.critical("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                logging.critical("\n\n")
                logging.critical("Iter: %d / %d, Mean_LOSS: %.4f,  Mean SSIM: %.4f, Mean PSNR: %.4f" %(n, args.iteration, mean_loss, mean_ssim, mean_psnr))
                logging.critical("\n\n")
                logging.critical(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                if mean_psnr < 17:
                    break
if __name__ == '__main__':
    main()