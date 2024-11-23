import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
from utils import split_data, timeSince
from model import bank_generator, cifar_generator
import argparse
import time
import os
from datasets import ExperimentDataset
from model import bank_generator, bank_net
import copy
from vfl import VFLNN, Client, Server

### This attack method is as follows:
#   1. Split the dataset into three parts for training, testing, and prediction
#   2. Train a classification model (logistic regression, random forest or neural network) using train and test data
#   3. Train a generator based on the trained classifier in Step 2 and the prediction dataset
#   4. compute overall mse 

def getSplittedDataset(trainpart, testpart, expset):
    x, y=expset[0]
    logging.critical("\n[FUNCTION]: Splitting dataset by getSplittedDataset()......")
    logging.info("Display first (x, y) pair of dataset:\n %s, %s", x, y)
    logging.info("Shape of (x, y): %s %s", x.shape, y.shape)
 
    train_len = int(len(expset) * trainpart)
    test_len = int(len(expset) * testpart)
    total_len = int(len(expset))
    # 训练集和测试集
    trainset, remainset = torch.utils.data.random_split(expset, [train_len, total_len-train_len])
    testset, predictset = torch.utils.data.random_split(remainset, [test_len, len(remainset)-test_len])
    logging.critical("len(trainset): %d", len(trainset))
    logging.critical("len(testset): %d", len(testset))
    logging.critical("len(predictset): %d", len(predictset))
    return trainset, testset, predictset
    

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
    
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.CRITICAL) # 只有critical级别的才会输出到控制台
    # add formatter to ch
    ch.setFormatter(logging.Formatter('%(message)s')) # 控制台只输出日志消息的内容
    logging.getLogger().addHandler(ch) 

def vflnn_train(target_vflnn, train_loader, test_loader, device, args):

    for epoch in range(args.epochs_train):
        target_vflnn.train()
        train_loss = 0

        for batchidx, (data, target) in enumerate(train_loader):
            data, target_label = data.to(device), target.to(device)
            target_vflnn.zero_grads()
            x_a, x_b = split_data(data, args.dataset)
            target_vflNN_output = target_vflnn(x_a, x_b)
            # 计算loss
            target_vflNN_loss = F.cross_entropy(target_vflNN_output, target_label.long())
            # 反向传播
            target_vflNN_loss.backward()
            # 整体vflNN的反向传播
            target_vflnn.backward()

            train_loss += target_vflNN_loss.item() * data.size(0)
            # 更新模型
            target_vflnn.step()
        # 每5轮测试test一次
        if epoch % 5 == 0 or epoch == args.epochs_train - 1:
            correct = 0
            total = 0
            with torch.no_grad():
                for data, label in test_loader:
                    data, label = data.to(device), label.to(device)
                    x_a, x_b = split_data(data, args.dataset)
                    output = target_vflnn(x_a, x_b)
                    _, predicted = torch.max(output.data, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item() 
            acc = 100 * correct / total
            logging.critical('Epoch: {} | Loss: {:.6f} | Test Accuracy: {:.3f}'.format(epoch, train_loss, acc))

    

# 训练生成器 output_dim 是要恢复的特征数量(被动方特征数) input_dim - output_dim = 主动方特征数
class GeneratorTrainer():
    def __init__(self, input_dim, output_dim, args, device):
        super().__init__()
        logging.critical("\n[FUNCTION]: Initializing GeneratorTrainer......")
        logging.critical("Creating GeneratorTrainer with input_dim: %d, output_dim: %d", input_dim, output_dim)

        self.netG = bank_generator(input_dim, output_dim).to(device)
        logging.info("Structure of Generator: %s", self.netG)
        self.device = device
        self.args = args
        self.pas_feature = output_dim
        self.act_feature = input_dim - output_dim

    def train(self, target_vflnn, predict_loader, enableMean, mean_feature):
        logging.critical("\n[FUNCTION]: Training Generator.......")
        target_vflnn.eval()
        log_interval = self.args.epochs_attack //5
        # self.G是特征生成器
        self.netG.train()
        optimizerG = optim.Adam(self.netG.parameters(), lr=0.0001)

        for epoch in range(self.args.epochs_attack):
            acc = 0.0
            losses = []
            for x, y in predict_loader:
                optimizerG.zero_grad()
                x, y = x.to(self.device), y.to(self.device)
                # x_a 是要恢复的输入， x_a 是被动方底部模型的输入，x_b是主动方底部模型的输入
                x_a, x_b = split_data(x, self.args.dataset)
                # 生成器的输入
                noise = torch.randn(x_a.shape).to(self.device)
                fake_netG_input = torch.cat((noise, x_b), dim = 1)
                # 恢复的部分特征
                rec_feature = self.netG(fake_netG_input)
                # 将恢复的特征和原始特征x_b拼接
                y_final = target_vflnn(rec_feature, x_b)

                y_truth = target_vflnn(x_a, x_b)

                mean_loss = 0
                unknown_var_loss = 0
                
                for i in range(rec_feature.size(1)):
                    if enableMean:
                        mean_loss = mean_loss + (rec_feature[:, i].mean() - mean_feature[i]) ** 2
                    # rec_feature第i列的方差
                    unknown_var_loss = unknown_var_loss + (rec_feature[:, i].var())

                loss = ((y_final - y_truth.detach()) ** 2).sum() + self.args.meanLambda * mean_loss + self.args.unknownVarLambda * unknown_var_loss
                loss.backward()
                losses.append(loss.detach())
                # Update the generator
                optimizerG.step()

            if epoch % log_interval == 0 or epoch == self.args.epochs_attack - 1:
                logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>')
                logging.critical('Epoch: %d, Loss: %s', epoch, sum(losses) / len(losses))
                logging.info('L2 norm of rec_feature: %s', (rec_feature**2).sum())
                logging.info("L2 norm of original vector: %s ", (x_a ** 2).sum())
                logging.info("First two lines of rec_feature: %s", rec_feature[:2, :])

    def test(self, predictloader, mean_feature):
        # 计算单个特征的mse
        def loss_per_feature(recover, target):
            res = []
            for i in range(recover.size(1)):
                loss = ((recover[:, i] - target[:, i]) ** 2).mean().item()
                res.append(loss)
            return np.array(res) # 返回一个数组，每个特征的损失

        logging.critical("\n[FUNCTION]: Testing Generator.......")
        self.netG.eval()

        mse = torch.nn.MSELoss(reduction='mean')
        generator_loss = []
        total_per_feature_loss = None
        output = 10

        for x, y in predictloader:
            x, y = x.to(self.device), y.to(self.device)
            x_a, x_b = split_data(x, self.args.dataset)
            noise = torch.randn(x_a.shape).to(self.device)
            fake_netG_input = torch.cat((noise, x_b), dim=1)
            # 恢复的被动方的特征
            rec_feature = self.netG(fake_netG_input)

            model_loss = mse(rec_feature, x_a).item()
            generator_loss.append(model_loss)

            model_loss_per_feature = loss_per_feature(x_a, rec_feature)
            total_per_feature_loss = model_loss_per_feature if total_per_feature_loss is None else total_per_feature_loss + model_loss_per_feature

        
            logging.critical("<<<<<<<<<<<<<<<<<<<<<")
            logging.critical("Model Loss: %s", model_loss)
            logging.critical("Model Loss per feature: %s", model_loss_per_feature)
            logging.critical("First two lines of rec_feature: %s", rec_feature[:2, :])
            logging.critical("First two lines of x_a: %s", x_a[:2, :])
        
        
        logging.critical("----------------------SUMMARY----------------------")
        mean_model_moss = sum(generator_loss) / len(generator_loss)
        mean_per_feature_loss = total_per_feature_loss / len(generator_loss)

        logging.critical("Mean Model Loss: %s", mean_model_moss)
        logging.critical("Mean Feature Loss: %s", mean_per_feature_loss)
        return mean_model_moss
    

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description='GRNN attack')
    parser.add_argument('--dataset', type=str, default='bank', help='dataset to use')
    parser.add_argument('--epochs_train', type=int, default=60, help='number of epochs to train')
    parser.add_argument('--epochs_attack', type=int, default=60, help='number of epochs to predict')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--meanLambda', type=float, default=1.2, help='lambda for mean loss')
    parser.add_argument('--unknownVarLambda', type=float, default=0.25, help='lambda for unknown var loss')
    parser.add_argument('--train_portion', type=float, default=0.6, help='portion of training data')
    parser.add_argument('--test_portion', type=float, default=0.2, help='portion of testing data')  
    parser.add_argument('--predict_portion', type=float, default=0.2, help='portion of prediction data')
    parser.add_argument('--backup', type=bool, default=True, help='record the log')
    
    args = parser.parse_args()

    if args.dataset == 'bank':
        g_input_dim = 20
        g_output_dim = 10
        classfier_input_dim = 10
        classfier_ouput_dim = 2
        dataset_path = 'data/bank_cleaned.csv'
    elif args.dataset == 'drive':
        g_input_dim = 48
        g_output_dim = 24
        classfier_input_dim = 24
        classfier_ouput_dim = 11
        dataset_path = 'data/drive_cleaned.csv'

    enableMean = False
     


    data_time_fime = time.strftime("%Y-%m-%d-%H", time.localtime())

    if args.backup:
        path_name = os.path.join('log', 'GRNA', args.dataset)
        os.makedirs(path_name, exist_ok=True)
        initlogging(logfile=os.path.join(path_name, data_time_fime + '.log'))
        logging.info(">>>>>>>>>>>>>>Running Settings>>>>>>>>>>>>>>")
        for arg in vars(args):
            logging.info("%s: %s", arg, getattr(args, arg))
        logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n")

    
    expset = ExperimentDataset(datafilepath=dataset_path)
    logging.critical("For dataset %s, dataset length: %d", args.dataset, len(expset))

    start = time.time()
    trainset, testset, predictset = getSplittedDataset(args.train_portion, args.test_portion, expset)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)
    predictloader = torch.utils.data.DataLoader(predictset, batch_size=args.batch_size, shuffle=True)
    logging.info("len(trainloader): %d", len(trainloader))
    logging.info("len(testloader): %d", len(testloader))
    logging.info("len(predictloader): %d", len(predictloader))

    bottom_model1, top_model = bank_net(classfier_input_dim, classfier_ouput_dim)
    bottom_model2 = copy.deepcopy(bottom_model1)
    bottom_model1, bottom_model2, top_model = bottom_model1.to(device), bottom_model2.to(device), top_model.to(device)


    pas_client_optimizer = optim.Adam(bottom_model1.parameters(), lr=0.001)
    act_client_optimizer = optim.Adam(bottom_model2.parameters(), lr=0.001)
    server_optimizer = optim.Adam(top_model.parameters(), lr=0.001)


    target_vflnn = VFLNN(Client(bottom_model1), Client(bottom_model2), Server(top_model, 1), [pas_client_optimizer, act_client_optimizer],server_optimizer)

    generatorTrainer = GeneratorTrainer(g_input_dim, g_output_dim, args, device)

    # 先训练全局vfl模型
    vflnn_train(target_vflnn, trainloader, testloader, device, args)
    # 训练生成器
    generatorTrainer.train(target_vflnn, predictloader, enableMean, expset.mean_attr)

    # 测试生成器
    mean_model_loss = generatorTrainer.test(predictloader, expset.mean_attr)

    logging.critical('%s' % (timeSince(start)))
    logging.critical("Mean Model Loss: %s", mean_model_loss)

    


