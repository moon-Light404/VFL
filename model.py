import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, bn=False, stride=1):
        super(ResBlock, self).__init__()
        self.bn = bn
        if bn:
            self.bn0 = nn.BatchNorm2d(in_planes)

        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        if bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1)
        self.shortcut = nn.Sequential()

        if stride > 1:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn0(x))
        else:
            out = F.relu(x)

        if self.bn:
            out = F.relu(self.bn1(self.conv1(out)))
        else:
            out = F.relu(self.conv1(out))

        out = self.conv2(out)
        out += self.shortcut(x)
        return out

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False), # 深度可分离卷积
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False), # 1x1卷积
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


def cifar_mobilenet(level):
    client = []
    server = []

    if level == 1:
        client += conv_bn(  3,  32, 2)
    #     client += nn.Sequential(
    #     nn.Conv2d(3,32, 3, 2, 1, bias=False),
    #     nn.ReLU(inplace=True)
    # )
        server += conv_dw( 32,  64, 1)
        server += conv_dw( 64, 128, 2)
        server += conv_dw(128, 128, 1)
        server += conv_dw(128, 256, 2)
        server += conv_dw(256, 256, 1)
        server += conv_dw(256, 512, 2)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 1024, 2)
        server += conv_dw(1024, 1024, 1)
        server += nn.Sequential(nn.AvgPool2d(1),nn.Flatten(),nn.Linear(1024, 10))
        return nn.Sequential(*client),nn.Sequential(*server)
    if level == 2:
        client += conv_bn(  3,  32, 2)
        client += conv_dw( 32,  64, 1)

        server += conv_dw( 64, 128, 2)
        server += conv_dw(128, 128, 1)
        server += conv_dw(128, 256, 2)
        server += conv_dw(256, 256, 1)
        server += conv_dw(256, 512, 2)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 1024, 2)
        server += conv_dw(1024, 1024, 1)
        server += nn.Sequential(nn.AvgPool2d(1),nn.Flatten(),nn.Linear(1024, 10))
        return nn.Sequential(*client),nn.Sequential(*server)
    if level == 3:
        client += conv_bn(  3,  32, 2)
        client += conv_dw( 32,  64, 1)
        client += conv_dw( 64, 128, 2)
        client += conv_dw(128, 128, 1)

        server += conv_dw(128, 256, 2)
        server += conv_dw(256, 256, 1)
        server += conv_dw(256, 512, 2)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 1024, 2)
        server += conv_dw(1024, 1024, 1)
        server += nn.Sequential(nn.AvgPool2d(1),nn.Flatten(),nn.Linear(1024, 10))
        return nn.Sequential(*client),nn.Sequential(*server)
    if level == 4:
        client += conv_bn(  3,  32, 2)
        client += conv_dw( 32,  64, 1)
        client += conv_dw( 64, 128, 2)
        client += conv_dw(128, 128, 1)
        client += conv_dw(128, 256, 2)
        client += conv_dw(256, 256, 1)

        server += conv_dw(256, 512, 2)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 1024, 2)
        server += conv_dw(1024, 1024, 1)
        server += nn.Sequential(nn.AvgPool2d(1),nn.Flatten(),nn.Linear(1024, 10))
        return nn.Sequential(*client),nn.Sequential(*server)

def vgg16_make_layers(cfg, batch_norm=True, in_channels=3):
    layers = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def vgg16(level, batch_norm, num_class = 10):

    client_net = []
    server_net = []
    # print(level)
    if level == 1 :
        client_net += vgg16_make_layers([32, 32, "M"], batch_norm, in_channels=3)
        server_net += vgg16_make_layers([128, 128, "M"], batch_norm, in_channels=64)
        server_net += vgg16_make_layers([256, 256, 256, "M"], batch_norm, in_channels=128)
        server_net += vgg16_make_layers([512, 512, 512, "M"], batch_norm, in_channels=256)
        server_net += vgg16_make_layers([512, 512, 512, "M"], batch_norm, in_channels=512)
        server_net += [nn.AdaptiveAvgPool2d((1, 1))]
        server_net += [nn.Flatten(),nn.Linear(512 * 1 * 1, num_class)]
        return nn.Sequential(*client_net),nn.Sequential(*server_net)


    if level == 2 :
        client_net += vgg16_make_layers([64,64,"M"], batch_norm, in_channels=3)
        server_net += vgg16_make_layers([128, 128, "M"], batch_norm, in_channels=64)
        server_net += vgg16_make_layers([256, 256, 256, "M"], batch_norm, in_channels=128)
        server_net += vgg16_make_layers([512, 512, 512, "M"], batch_norm, in_channels=256)
        server_net += vgg16_make_layers([512, 512, 512, "M"], batch_norm, in_channels=512)
        server_net += [nn.AdaptiveAvgPool2d((1, 1))]
        server_net += [nn.Flatten(),nn.Linear(512 * 1 * 1, num_class)]
        return nn.Sequential(*client_net),nn.Sequential(*server_net)

    if level == 3:
        client_net += vgg16_make_layers([64, 64, "M"], batch_norm, in_channels=3)
        client_net += vgg16_make_layers([128, 128, "M"], batch_norm, in_channels=64)
        server_net += vgg16_make_layers([256, 256, 256, "M"], in_channels=128)
        server_net += vgg16_make_layers([512, 512, 512, "M"], in_channels=256)
        server_net += vgg16_make_layers([512, 512, 512, "M"], in_channels=512)
        server_net += [nn.AdaptiveAvgPool2d((1, 1))]
        server_net += [nn.Flatten(),nn.Linear(512 * 1 * 1, num_class)]
        return nn.Sequential(*client_net),nn.Sequential(*server_net)

    if level == 4:
        client_net += vgg16_make_layers([64, 64, "M"], batch_norm, in_channels=3)
        client_net += vgg16_make_layers([128, 128, "M"], batch_norm, in_channels=64)
        client_net += vgg16_make_layers([256, 256, 256, "M"], batch_norm, in_channels=128)
        server_net += vgg16_make_layers([512, 512, 512, "M"], in_channels=256)
        server_net += vgg16_make_layers([512, 512, 512, "M"], in_channels=512)
        server_net += [nn.AdaptiveAvgPool2d((1, 1))]
        server_net += [nn.Flatten(),nn.Linear(512 * 1 * 1, num_class)]
        return nn.Sequential(*client_net),nn.Sequential(*server_net)
    
# cifar_decoder 是将被动方和主动方的输出拼接特征||-->恢复为原始图片
def cifar_decoder(input_shape, level, channels=3):
    
    net = []
    #act = "relu"
    act = None
    print("[DECODER] activation: ", act)

    net += [nn.ConvTranspose2d(input_shape[0], 256, 3, 2, 1, output_padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True)]

    if level <= 2:
        net += [nn.Conv2d(256, channels, 3, 1, 1), nn.BatchNorm2d(channels)]
        net += [nn.Tanh()]
        return nn.Sequential(*net)
    
    net += [nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True)]

    if level == 3:
        net += [nn.Conv2d(128, channels, 3, 1, 1), nn.BatchNorm2d(channels)]
        net += [nn.Tanh()]
        return nn.Sequential(*net)
    
    net += [nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True)]

    if level == 4:
        net += [nn.Conv2d(64, channels, 3, 1, 1), nn.BatchNorm2d(channels)]
        net += [nn.Tanh()]
        return nn.Sequential(*net)
    
def cifar_discriminator_model(input_shape, level, agn=False, fc_dim=512):

    net = []
    if level <=2:
        net += [nn.Conv2d(input_shape, 128, 3, 2, 1)] # inchannels = input_shape[0]
        net += [nn.ReLU()]
        net += [nn.Conv2d(128, 256, 3, 2, 1)]
    elif level == 3:
        net += [nn.Conv2d(input_shape, 256, 3, 2, 1)]
    elif level == 4:
        net += [nn.Conv2d(input_shape, 256, 3, 1, 1)]


    bn = False
        
    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]

    net += [nn.Conv2d(256, 256, 3, 2, 1)]
    net += [nn.Flatten()]
    # 修改1024--> 512
    # net += [nn.Linear(512, 1)] # vfl分割后的一半特征
    # 区分图片特征
    if agn == True:
        net += [nn.Linear(4096, 1024)]
        net += [nn.Linear(1024, 1)]
    else:
        net += [nn.Linear(fc_dim, 1)] # cifar_model 512
    return nn.Sequential(*net)

def cifar_pseudo(level):
    client = []
    if level == 1:
        client += conv_bn(  3,  32, 2)
        client += conv_dw(  32,  32, 1)
 
        return nn.Sequential(*client)
    if level == 2:
        client += conv_bn(  3,  32, 2)
        client += conv_dw( 32,  64, 1)
        client += conv_dw( 64,  64, 1)
        return nn.Sequential(*client)
    if level == 3:
        client += conv_bn(  3,  32, 2)
        client += conv_dw( 32,  64, 1)
        client += conv_dw( 64, 128, 2)
        client += conv_dw(128, 128, 1)
        client += conv_dw(128, 128, 1)
        return nn.Sequential(*client)
    if level == 4:
        client += conv_bn(  3,  32, 2)
        client += conv_dw( 32,  64, 1)
        client += conv_dw( 64, 128, 2)
        client += conv_dw(128, 128, 1)
        client += conv_dw(128, 256, 2)
        client += conv_dw(256, 256, 1)
        client += conv_dw(256, 256, 1)
        return nn.Sequential(*client)
    
def bank_net(input_dim, output_dim):
    client = []
    server = []
    # client 输入10 输出10
    client += [nn.Linear(input_dim, 200)]
    client += [nn.ReLU(inplace=True)]
    client += [nn.Linear(200, 100)]
    client += [nn.ReLU(inplace=True)]
    client += [nn.Linear(100, 30)]

    server += [nn.Linear(60, 120)]
    server + [nn.ReLU(inplace=True)]
    server += [nn.Linear(120, 50)]
    server += [nn.ReLU(inplace=True)]
    server += [nn.Linear(50, output_dim)]
    return nn.Sequential(*client), nn.Sequential(*server)


def bank_decoder(input_dim, output_dim):
    net = []
    net += [nn.Linear(input_dim, 200)]
    net += [nn.ReLU(inplace=True)]
    net += [nn.Linear(200, 100)]
    net += [nn.ReLU(inplace=True)]
    net += [nn.Linear(100, output_dim)]
    return nn.Sequential(*net)

def bank_pseudo(input_dim, output_dim):
    net = []
    net += [nn.Linear(input_dim, 100)]
    net += [nn.ReLU(inplace=True)]
    net += [nn.Linear(100, 50)]
    net += [nn.ReLU(inplace=True)]
    net += [nn.Linear(50, output_dim)]
    return nn.Sequential(*net)


def bank_discriminator(input_dim, output_dim=1):
    net = []
    net += [nn.Linear(input_dim, 200)]
    net += [nn.ReLU(inplace=True)]
    net += [nn.Linear(200, 100)]
    net += [nn.ReLU(inplace=True)]
    net += [nn.Linear(100, output_dim)]
    return nn.Sequential(*net)


# 生成taregt feature的生成器
class cifar_generator(nn.Module):
    def __inti__(self):
        super(cifar_generator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),  # (16, 32, 32)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))  # (16, 32, 16)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),  # (3, 32, 16)
            nn.BatchNorm2d(3),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    
class bank_generator(nn.Module):
    def __init__(self, latent_dim, target_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 200), 
            nn.ReLU(),
            nn.Linear(200, 100), 
            nn.ReLU(),
            nn.Linear(100, target_dim),
        )
    def forward(self, x):
        return self.net(x)


def Resnet(level, output_dim = 100):
    client = []
    server = []
    bn = False
    if level == 1:
        client += nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                nn.BatchNorm2d(64),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
       
        server += [ResBlock(64, 64, bn=bn)]
        server += [ResBlock(64, 64, bn=bn)]
        server += [ResBlock(64, 128, bn=bn, stride=2)]
        server += [ResBlock(128, 128, bn=bn)]
        server += [ResBlock(128, 256, bn=bn,stride=2)]
        server += [ResBlock(256, 256, bn=bn)]
        server += [ResBlock(256, 512, bn=bn,stride=2)]
        server += [ResBlock(512, 512, bn=bn)]
        server += nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten(),
                                nn.Linear(512, output_dim)
        )
        return nn.Sequential(*client), nn.Sequential(*server)
    if level == 2:
        client += nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(64),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        client += [ResBlock(64, 64, bn=bn)]
        client += [ResBlock(64, 64, bn=bn)]
        server += [ResBlock(64, 128, bn=bn, stride=2)]
        server += [ResBlock(128, 128, bn=bn)]
        server += [ResBlock(128, 256, bn=bn,stride=2)]
        server += [ResBlock(256, 256, bn=bn)]
        server += [ResBlock(256, 512, bn=bn,stride=2)]
        server += [ResBlock(512, 512, bn=bn)]
        server += nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten(),
                                nn.Linear(512, output_dim)
        )
        return nn.Sequential(*client), nn.Sequential(*server)
    if level == 3:
        client += nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(64),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        client += [ResBlock(64, 64, bn=bn)]
        client += [ResBlock(64, 64, bn=bn)]
        client += [ResBlock(64, 128, bn=bn, stride=2)]
        client += [ResBlock(128, 128, bn=bn)]
        server += [ResBlock(128, 256, bn=bn,stride=2)]
        server += [ResBlock(256, 256, bn=bn)]
        server += [ResBlock(256, 512, bn=bn,stride=2)]
        server += [ResBlock(512, 512, bn=bn)]
        server += nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten(),
                                nn.Linear(512, output_dim)
        )
        return nn.Sequential(*client), nn.Sequential(*server)
    if level == 4:
        client += nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                nn.BatchNorm2d(64),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        client += [ResBlock(64, 64, bn=bn)]
        client += [ResBlock(64, 64, bn=bn)]
        client += [ResBlock(64, 128, bn=bn, stride=2)]
        client += [ResBlock(128, 128, bn=bn)]
        client += [ResBlock(128, 256, bn=bn,stride=2)]
        client += [ResBlock(256, 256, bn=bn,)]
        server += [ResBlock(256, 512, bn=bn,stride=2)]
        server += [ResBlock(512, 512, bn=bn,)]
        server += nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten(),
                                nn.Linear(512, output_dim)
        )
        return nn.Sequential(*client), nn.Sequential(*server)


def resnet_pseudo(level = 2):
    client = []
    return client

def resent_decoder(input_shape, level, channels=3):
    net = []
    #act = "relu"
    act = None
    print("[DECODER] activation: ", act)
    # 第一层采样 16x16 -> 32x32 ||  8x8 -> 16x16 ||   4x4 -> 8x8
    net += [nn.ConvTranspose2d(input_shape[0], 256, 3, 2, 1, output_padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True)]

    if level <= 2:
        # 第二层上采样：32x32 -> 64x64
        net += [nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True)]
        # 第三层上采样：128 -> 3 通道，保持空间尺寸
        net += [nn.Conv2d(128, channels, 3, 1, 1), nn.BatchNorm2d(channels)]
        net += [nn.Tanh()]
        return nn.Sequential(*net)
    # 第二层上采样：16x16 -> 32x32  8x8 -> 16x16
    net += [nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True)]

    if level == 3:
        # 第三层上采样：32x32 -> 64x64
        net += [nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True)]
        # 第四层上采样：64 -> 3 -> 通道，保持空间尺寸
        net += [nn.Conv2d(64, channels, 3, 1, 1), nn.BatchNorm2d(channels)]
        net += [nn.Tanh()]
        return nn.Sequential(*net)
    # 第三层上采样：16x16 -> 32x32 
    net += [nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True)]

    if level == 4:
        # 第四层上采样：32x32 -> 64x64
        net += [nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True)]
        # 第五层上采样：64 -> 3 -> 通道，保持空间
        net += [nn.Conv2d(32, channels, 3, 1, 1), nn.BatchNorm2d(channels)]
        net += [nn.Tanh()]
        return nn.Sequential(*net)

def resnet_discriminator(input_shape, level):
    net = []
    if level <= 2:
        net += [nn.Conv2d(input_shape, 128, 3, 2, 1)] 
        net += [nn.ReLU()]
        net += [nn.Conv2d(128, 256, 3, 2, 1)]
        net += [nn.ReLU()]
        net += [nn.Conv2d(256, 256, 3, 2, 1)]
    elif level == 3:
        net += [nn.Conv2d(input_shape, 256, 3, 2, 1)]
        net += [nn.ReLU()]
        net += [nn.Conv2d(256, 256, 3, 2, 1)]
    elif level == 4:
        net += [nn.Conv2d(input_shape, 256, 3, 2, 1)]
        net += [nn.ReLU()]
        net += [nn.Conv2d(256, 256, 3, 2, 1)]

    bn = False
    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]

    net += [nn.Conv2d(256, 256, 3, 2, 1)]
    net += [nn.Flatten()]
    net += [nn.Linear(256, 1)]
    return nn.Sequential(*net)
