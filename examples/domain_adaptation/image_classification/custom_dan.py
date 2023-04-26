"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import argparse
import datetime
import gc
import os
import os.path as osp
import random
import shutil
import time
import warnings
from warnings import warn

import custom_utils
import cv2
import matplotlib
import matplotlib.colors as col
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from custom_utils import plot_graph
from matplotlib import pyplot as plt
from numpy.linalg import LinAlgError, linalg, solve
from scipy.stats import chi2
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay
from torch.distributions import Chi2
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary

from tllib.alignment.dan import (ImageClassifier,
                                 MultipleKernelMaximumMeanDiscrepancy)
from tllib.modules.kernels import GaussianKernel
from tllib.utils.analysis import a_distance, collect_feature, tsne
from tllib.utils.data import ForeverDataIterator
from tllib.utils.logger import CompleteLogger
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.metric import accuracy

gc.collect()
torch.cuda.empty_cache()
warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""


def tsne_visualize(source_feature: torch.Tensor, target_feature: torch.Tensor, filename: str, source_color='r', target_color='b'):
    """
    Visualize features from different domains using t-SNE.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'

    """
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    features = np.concatenate([source_feature, target_feature], axis=0)
    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

    # domain labels, 1 represents source while 0 represents target
    domains = np.concatenate(
        (np.ones(len(source_feature)), np.zeros(len(target_feature))))

    # create color map
    cmap = matplotlib.colors.ListedColormap([target_color, source_color])

    # create legend handles
    legend_handles = [matplotlib.patches.Patch(color=target_color, label='Target feature'),
                      matplotlib.patches.Patch(color=source_color, label='Source feature')]

    # visualize using matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # scatter plot
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=cmap, s=20)

    # add legend
    plt.legend(handles=legend_handles)

    # hide x and y ticks
    plt.xticks([])
    plt.yticks([])

    # save figure
    plt.savefig(filename)

def mahalanobis_distance(difference, num_random_features):
    num_samples, _ = difference.shape
    sigma = torch.cov(difference.T)

    # try:
    #     torch.linalg.pinv(sigma)
    # except LinAlgError:
    #     warn('covariance matrix is singular. Pvalue returned is 1.1')
    #     raise

    mu = torch.mean(difference, 0)

    if num_random_features == 1:
        stat = float(num_samples * torch.pow(mu,2)) / float(sigma)
    else:
        sigma = torch.pinverse(sigma)
        stat = num_samples * torch.matmul(mu, torch.matmul(sigma, mu.T))

    return chi2.sf(stat.detach().cpu(), num_random_features)

# def mahalanobis_distance(difference, num_random_features):
#     num_samples, _ = shape(difference)
#     sigma = cov(transpose(difference))

#     try:
#         linalg.inv(sigma)
#     except LinAlgError:
#         warn('covariance matrix is singular. Pvalue returned is 1.1')
#         raise

#     mu = mean(difference, 0)

#     if num_random_features == 1:
#         stat = float(num_samples * mu ** 2) / float(sigma)
#     else:
#         stat = num_samples * mu.dot(solve(sigma, transpose(mu)))

#     return chi2.sf(stat, num_random_features)

class MeanEmbeddingTest:

    def __init__(self, data_x, data_y, scale, number_of_random_frequencies, device):
        self.device = device
        self.data_x = scale * data_x.to(device)
        self.data_y = scale * data_y.to(device)
        self.number_of_frequencies = number_of_random_frequencies
        self.scale = scale

    def get_estimate(self, data, point):
        z = data - self.scale * point
        z2 = torch.norm(z, p=2, dim=1)**2
        return torch.exp(-z2/2.0)


    def get_difference(self, point):
        return self.get_estimate(self.data_x, point) - self.get_estimate(self.data_y, point)


    def vector_of_differences(self, dim):
        points = torch.randn(self.number_of_frequencies,
                             dim, device=self.device)
        a = [self.get_difference(point) for point in points]
        return torch.stack(a).T
    
    def compute_pvalue(self):

        _, dimension = self.data_x.size()
        obs = self.vector_of_differences(dimension)

        return mahalanobis_distance(obs, self.number_of_frequencies)

def split_data(df, frac=0.2):
    seletected = df['flow_id'].drop_duplicates().sample(frac=frac)

    val = df[df['flow_id'].isin(seletected)]
    train = df[~df['flow_id'].isin(seletected)]
    return train, val


def source_target_split(df, choice, frac=0.5):
    df_label_choice = df[df.Label == choice]
    print("Selected label " + str(choice))
    seletected_label_3 = df_label_choice['flow_id'].drop_duplicates().sample(
        frac=0.99)
    seletected = df['flow_id'].drop_duplicates().sample(frac=frac)

    source_select = seletected[~seletected.isin(seletected_label_3)]

    source = df[df['flow_id'].isin(source_select)]
    target = df[~df['flow_id'].isin(seletected)]
    return source, target


def resize_image(image, target_size=(224, 224)):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def most_frequent(List):
    return max(set(List), key=List.count)


def data_processing(raw_data):
    # Get flow label
    result = raw_data.groupby('flow_id')['Label'].apply(list).to_dict()
    flow_label = []
    for flow in result:
        flow_label.append(most_frequent(result[flow]))
    flow_label = np.array(flow_label)
    # Reshape payloads
    true_data = raw_data.drop('flow_id', axis=1)
    datas = true_data.drop('Label', axis=1).to_numpy()/255
    datas = datas.reshape(-1, 20, 256).astype('float32')
    # Resize each image in the dataset
    datas = np.array([resize_image(img) for img in datas])
    rgb_datas = np.repeat(datas[:, :, np.newaxis, ], 3, axis=2)
    rgb_datas = np.moveaxis(rgb_datas, 2, 1)
    final_dataset = MyDataset(rgb_datas, flow_label)
    return final_dataset


def remapping(df, map):
    df_copy = df.copy()
    df_copy['Label'] = df_copy['Label'].replace(map)
    return df_copy


class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    if args.data == 'GQUIC':
        print('GQUIC data')
        args.class_names = ['File_transfer', 'Music', 'VoIP', 'Youtube']
        num_classes = 4

        # Set data path
        Train_path = '/home/bkcs/Transfer-Learning-Library/examples/domain_adaptation/image_classification/data/gquic/Train/GQUIC_data_256.feather'
        Test_path = '/home/bkcs/Transfer-Learning-Library/examples/domain_adaptation/image_classification/data/gquic/Test/GQUIC_test_256.feather'
        fallback_Train_path = '/home/bkcs/HDD/Transfer-Learning-Library/examples/domain_adaptation/image_classification/data/gquic/Train/GQUIC_data_256.feather'
        fallback_Test_path = '/home/bkcs/HDD/Transfer-Learning-Library/examples/domain_adaptation/image_classification/data/gquic/Test/GQUIC_test_256.feather'

        if os.path.isfile(Train_path):
            Train_data = pd.read_feather(Train_path)
            Test_data = pd.read_feather(Test_path)
        else:
            print("File not found at path:", Train_path)
            print("Change to:", fallback_Train_path)
            Train_data = pd.read_feather(fallback_Train_path)
            Test_data = pd.read_feather(fallback_Test_path)
        train_source, train_target = source_target_split(
            Train_data, choice=args.label)
        train_target, val_raw = split_data(train_target)
        test_raw = Test_data
        del Train_data
        del Test_data
    elif args.data == 'Capture':
        print('Capture data')
        args.class_names = ['Ecommerce', 'Video']
        num_classes = 2

        # Set data path
        Train_path = '/home/bkcs/HDD/FL/Data_Processing/Capture_small/Train/Capture_data_20_256.feather'
        Test_path = '/home/bkcs/HDD/FL/Data_Processing/Capture_small/Test/Capture_test_20_256.feather'
        fallback_Train_path = '/home/bkcs/HDD/FL/Data_Processing/Capture_small/Train/Capture_data_20_256.feather'
        fallback_Test_path = '/home/bkcs/HDD/FL/Data_Processing/Capture_small/Test/Capture_test_20_256.feather'

        if os.path.isfile(Train_path):
            Train_data = pd.read_feather(Train_path)
            Test_data = pd.read_feather(Test_path)
        else:
            print("File not found at path:", Train_path)
            print("Change to:", fallback_Train_path)
            Train_data = pd.read_feather(fallback_Train_path)
            Test_data = pd.read_feather(fallback_Test_path)

        label_mapping = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}

        source_labels = [1, 3, 5]
        target_labels = [2, 4]

        train_source = Train_data.loc[Train_data['Label'].isin(source_labels)]
        train_target = Train_data.loc[Train_data['Label'].isin(target_labels)]
        val_raw = Test_data.loc[Test_data['Label'].isin(source_labels)]
        test_raw = Test_data.loc[Test_data['Label'].isin(target_labels)]

        train_source = remapping(train_source, label_mapping)
        train_target = remapping(train_target, label_mapping)
        val_raw = remapping(val_raw, label_mapping)
        test_raw = remapping(test_raw, label_mapping)
    else:
        print('Concate data')
        args.class_names = ['Ecommerce', 'Video', 'Google_Service']
        num_classes = len(args.class_names)
        if args.kich_ban == "S2T":
            train_source = pd.read_feather(
                '/home/bkcs/HDD/Transfer-Learning-Library/examples/domain_adaptation/image_classification/data/concat/train_source.feather')
            train_target = pd.read_feather(
                '/home/bkcs/HDD/Transfer-Learning-Library/examples/domain_adaptation/image_classification/data/concat/train_target.feather')
            test_raw = val_raw = pd.read_feather(
                '/home/bkcs/HDD/Transfer-Learning-Library/examples/domain_adaptation/image_classification/data/concat/test_raw.feather')
        else:
            train_target = pd.read_feather(
                '/home/bkcs/HDD/Transfer-Learning-Library/examples/domain_adaptation/image_classification/data/concat/train_source.feather')
            train_source = pd.read_feather(
                '/home/bkcs/HDD/Transfer-Learning-Library/examples/domain_adaptation/image_classification/data/concat/train_target.feather')
            test_raw = val_raw = pd.read_feather(
                '/home/bkcs/HDD/Transfer-Learning-Library/examples/domain_adaptation/image_classification/data/concat/val_raw.feather')

    train_source_dataset = data_processing(train_source)
    train_target_dataset = data_processing(train_target)
    val_dataset = data_processing(val_raw)
    test_dataset = data_processing(test_raw)
    del train_source, train_target, val_raw, test_raw
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using model '{}'".format(args.arch))
    backbone = custom_utils.get_model(args.arch, pretrain=not args.scratch)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer, finetune=not args.scratch).to(device)
    # # print summary
    # print("Backbone")
    # print(summary(backbone, (3, 244, 244)))
    # print("Classifier")
    # print(summary(classifier, (3, 244, 244)))
    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters(), args.lr,
                    momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr *
                            (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # define loss function
    mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
        kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
        linear=not args.non_linear
    )

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(
            logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(
            classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
        source_feature = collect_feature(
            train_source_loader, feature_extractor, device)
        target_feature = collect_feature(
            train_target_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
        tsne_visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(
            source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1, loss1, scorema1, scoremi1, precisionma1, precisionmi1, recallma1, recallmi1, conf_mat, avg_time, min_time, max_time = custom_utils.validate(
            test_loader, classifier, args, device)
        print("Test result below...")
        print("test_acc1 = {:3.5f}".format(acc1))
        print("F1 macro = {:3.5f}".format(scorema1))
        print("F1 micro= {:3.5f}".format(scoremi1))
        print("precision macro= {:3.5f}".format(precisionma1))
        print("precision micro= {:3.5f}".format(precisionmi1))
        print("recall macro = {:3.5f}".format(recallma1))
        print("recall micro = {:3.5f}".format(recallmi1))
        print('avg_time = {:3.5f}'.format(avg_time))
        print('min_time = {:3.5f}'.format(min_time))
        print('max_time = {:3.5f}'.format(max_time))
        fig, ax = plt.subplots(figsize=(10, 10))
        conf_filename = osp.join(logger.visualize_directory, 'conf.pdf')
        disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_mat, display_labels=args.class_names)
        disp.plot(xticks_rotation='vertical', ax=ax, colorbar=False)
        plt.savefig(conf_filename, bbox_inches="tight")
        return

    # start training
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    best_acc1 = 0.
    start_time = time.time()
    for epoch in range(args.epochs):
        # train for one epoch
        train_acc1, train_loss1 = train(train_source_iter, train_target_iter, classifier, mkmmd_loss, optimizer,
                                        lr_scheduler, epoch, args)
        train_acc.append(train_acc1)
        train_loss.append(train_loss1)
        # evaluate on validation set
        acc1, loss1, scorema1, scoremi1, precisionma1, precisionmi1, recallma1, recallmi1, conf_mat, avg_time, min_time, max_time = custom_utils.validate(
            val_loader, classifier, args, device)
        val_acc.append(acc1)
        val_loss.append(loss1)
        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(),
                   logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'),
                        logger.get_checkpoint_path('best'))
        print(conf_mat)
        best_acc1 = max(acc1, best_acc1)
    # Calculate the elapsed time
    elapsed_time = time.time() - start_time

    # Convert elapsed time to hours, minutes, seconds, and smaller units of seconds
    hours, rem = divmod(elapsed_time, 3600)
    minutes, rem = divmod(rem, 60)
    seconds, microseconds = divmod(rem, 1)
    microseconds = round(microseconds, 3)

    # Print the elapsed time
    print("Elapsed time: {:0>2}:{:0>2}:{:05.3f}".format(
        int(hours), int(minutes), seconds + microseconds))
    print("best_acc1 = {:3.5f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1, loss1, scorema1, scoremi1, precisionma1, precisionmi1, recallma1, recallmi1, conf_mat, avg_time, min_time, max_time = custom_utils.validate(
        test_loader, classifier, args, device)
    print("Test result below...")
    print("test_acc1 = {:3.5f}".format(acc1))
    print("F1 macro = {:3.5f}".format(scorema1))
    print("F1 micro= {:3.5f}".format(scoremi1))
    print("precision macro= {:3.5f}".format(precisionma1))
    print("precision micro= {:3.5f}".format(precisionmi1))
    print("recall macro = {:3.5f}".format(recallma1))
    print("recall micro = {:3.5f}".format(recallmi1))
    print('avg_time = {:3.5f}'.format(avg_time))
    print('min_time = {:3.5f}'.format(min_time))
    print('max_time = {:3.5f}'.format(max_time))
    # graph
    fig = plt.figure(figsize=(10, 6))
    plot_graph(list(range(0, args.epochs)),
               train_acc, label='Train Accuracy')
    plot_graph(list(range(0, args.epochs)),
               val_acc, label='Val Accuracy')
    plt.legend()
    Accuracy_filename = osp.join(
        logger.visualize_directory, 'model_Accuracy.pdf')
    plt.savefig(Accuracy_filename, bbox_inches="tight")
    fig = plt.figure(figsize=(10, 6))
    plot_graph(list(range(0, args.epochs)),
               train_loss, label='Train Loss')
    plot_graph(list(range(0, args.epochs)),
               val_loss, label='Val Loss')
    plt.legend()
    Loss_filename = osp.join(logger.visualize_directory, 'model_Loss.pdf')
    plt.savefig(Loss_filename, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(10, 10))
    conf_filename = osp.join(logger.visualize_directory, 'conf.pdf')
    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_mat, display_labels=args.class_names)
    disp.plot(xticks_rotation='vertical', ax=ax, colorbar=False)
    plt.savefig(conf_filename, bbox_inches="tight")


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, model: ImageClassifier,
          mkmmd_loss: MultipleKernelMaximumMeanDiscrepancy, optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':5.4f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, trans_losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    mkmmd_loss.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)[:2]
        x_t, = next(train_target_iter)[:1]
        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_s, f_s = model(x_s)
        y_t, f_t = model(x_t)

        cls_loss = F.cross_entropy(y_s, labels_s)
        # print("f_s")
        # print(f_s)
        # print("f_t")
        # print(f_t)
        if args.loss_function == 'MKMMD':

            transfer_loss = mkmmd_loss(f_s, f_t)
        elif args.loss_function == "Both":
            mkme_loss = MeanEmbeddingTest(
                f_s, f_t, scale=args.scale_parameter, number_of_random_frequencies=args.random_frequencies, device=device)
            transfer_loss = (mkmmd_loss(f_s, f_t) +
                             mkme_loss.compute_pvalue())/2.0
        else:
            mkme_loss = MeanEmbeddingTest(
                f_s, f_t, scale=args.scale_parameter, number_of_random_frequencies=args.random_frequencies, device=device)
            transfer_loss = mkme_loss.compute_pvalue()
        # print(transfer_loss)
        loss = cls_loss + transfer_loss * args.trade_off

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        trans_losses.update(transfer_loss.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return cls_accs.avg, losses.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DAN for Unsupervised Domain Adaptation')
    # dataset parameters
    # parser.add_argument('root', metavar='DIR',
    #                     help='root path of dataset')
    # parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=utils.get_dataset_names(),
    #                     help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
    #                          ' (default: Office31)')
    parser.add_argument('-d', '--data', metavar='DATA', default='GQUIC',
                        choices=['GQUIC', 'Capture', 'Both'], help='Choice data')
    # parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    # parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    parser.add_argument('-l', '--label', type=int, default=3,
                        help="Label selected to reduce the number on source")
    # parser.add_argument('--train-resizing', type=str, default='default')
    # parser.add_argument('--val-resizing', type=str, default='default')
    # parser.add_argument('--resize-size', type=int, default=224,
    #                     help='the image size after resizing')
    # parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
    #                     help='Random resize scale (default: 0.08 1.0)')
    # parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
    #                     help='Random resize aspect ratio (default: 0.75 1.33)')
    # parser.add_argument('--no-hflip', action='store_true',
    #                     help='no random horizontal flipping during training')
    # parser.add_argument('--norm-mean', type=float, nargs='+',
    #                     default=(0.485, 0.456, 0.406), help='normalization mean')
    # parser.add_argument('--norm-std', type=float, nargs='+',
    #                     default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-lf', '--loss-function', metavar='LOSS FUNCTION',
                        default='MKMMD', help='loss function MK-MMD or MK-ME')
    parser.add_argument('-s_param', '--scale-parameter', type=float,
                        default=1, help='scale parameter of MK-ME loss function')
    parser.add_argument('-rf', '--random-frequencies', type=int,
                        default=5, help='number of random frequncies')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=custom_utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(custom_utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true',
                        help='whether train from scratch.')
    parser.add_argument('--non-linear', default=False, action='store_true',
                        help='whether not use the linear version')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.003, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0003,
                        type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75,
                        type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='custom_dan',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    parser.add_argument('-kb', '--kich-ban', metavar='KICh BAN',
                        default='S2T', help='Kich ban du lieu')
    args = parser.parse_args()
    main(args)
