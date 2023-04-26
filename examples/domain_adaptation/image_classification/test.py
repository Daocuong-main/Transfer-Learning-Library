"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import random
import time
import warnings
import argparse
import shutil
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

import utils
from tllib.alignment.dan import MultipleKernelMaximumMeanDiscrepancy, ImageClassifier
from tllib.modules.kernels import GaussianKernel
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import collect_feature, tsne, a_distance
import pandas as pd
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def split_data(df, frac=0.2):
    seletected = df['flow_id'].drop_duplicates().sample(frac=frac)
    val = df[df['flow_id'].isin(seletected)]
    train = df[~df['flow_id'].isin(seletected)]
    return train, val
def main():
    data = pd.read_feather('/media/bkcs/ea03a187-9ad2-44ad-9ec0-6246736e0fcd/gquic/Train/GQUIC_data_256.feather')
    test = pd.read_feather('/media/bkcs/ea03a187-9ad2-44ad-9ec0-6246736e0fcd/gquic/Test/GQUIC_test_256.feather')
    train_source_dataset = data[data.Label != 3]
    train_target_dataset, val_dataset = split_data(data[data.Label == 3])
    test_dataset = test[test.Label == 3]
    print(train_source_dataset.columns)
    print(test_dataset.columns)
    print("Success")
if __name__ == '__main__':
    main()