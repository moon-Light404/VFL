from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as tvmodels
from datetime import datetime
import logging

class ExperimentDataset(Dataset):
    def __init__(self, datafilepath):
        full_data_table = np.genfromtxt(datafilepath, delimiter=',')
        data = torch.from_numpy(full_data_table).float()
        self.samples = data[:, :-1] # 除了最后一列的数据
        batch, columns = self.samples.size()
        # permu_cols = torch.randperm(columns)
        permu_cols = range(columns)
        logging.critical("Dataset column permutation is: \n %s", permu_cols)
        self.samples = self.samples[:, permu_cols]
        
        self.labels = data[:, -1]
        min, _ = self.samples.min(dim=0)
        max, _ = self.samples.max(dim=0)
        self.feature_min = min
        self.feature_max = max
        # 对数据进行缩放0-1之间
        self.samples = (self.samples - self.feature_min)/(self.feature_max-self.feature_min)
        logging.critical("Creating dataset, len(samples): %d; positive labels sum: %d", len(self.labels), (self.labels > 0).sum().item())
        self.mean_attr = self.samples.mean(dim=0)
        self.var_attr = self.samples.var(dim=0) #s

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]
    

def getSplittedDataset(trainpart, testpart, expset):
    x, y=expset[0]
    logging.critical("\n[FUNCTION]: Splitting dataset by getSplittedDataset()......")
    logging.info("Display first (x, y) pair of dataset:\n %s, %s", x, y)
    logging.info("Shape of (x, y): %s %s", x.shape, y.shape)
 
    train_len = int(len(expset) * trainpart)
    test_len = int(len(expset) * testpart)
    total_len = int(len(expset))
    # 训练集和测试集
    trainset, testset = torch.utils.data.random_split(expset, [train_len, total_len-train_len])
    logging.critical("len(trainset): %d", len(trainset))
    
    logging.critical("len(testset): %d", len(testset))
    
    return trainset, testset