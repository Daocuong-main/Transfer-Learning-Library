"""
@author: Junguang Jiang, Baixu Chen
@contact: JiangJunguang1123@outlook.com, cbx_99_hasta@outlook.com
"""
import sys
import timm
from sklearn.metrics import precision_score, recall_score, f1_score
from matplotlib import pyplot as plt
from tllib.vision.datasets.imagelist import MultipleDomainsDataset
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.metric import accuracy, ConfusionMatrix
from tllib.vision.transforms import ResizeImage
import tllib.vision.models as models
import tllib.vision.datasets as datasets
import os.path as osp
import time
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from timm.data.auto_augment import auto_augment_transform, rand_augment_transform
from sklearn.utils.multiclass import type_of_target
sys.path.append('../../..')


def plot_graph(X, y, format='-', label=''):
    plt.plot(X, y, format, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(True)


def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()


def get_model(model_name, pretrain=True):
    if model_name in models.__dict__:
        # load models from tllib.vision.models
        backbone = models.__dict__[model_name](pretrained=pretrain)
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=pretrain)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    return backbone


def get_dataset_names():
    return sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    ) + ['Digits']


def get_dataset(dataset_name, root, source, target, train_source_transform, val_transform, train_target_transform=None):
    if train_target_transform is None:
        train_target_transform = train_source_transform
    if dataset_name == "Digits":
        train_source_dataset = datasets.__dict__[source[0]](osp.join(root, source[0]), download=True,
                                                            transform=train_source_transform)
        train_target_dataset = datasets.__dict__[target[0]](osp.join(root, target[0]), download=True,
                                                            transform=train_target_transform)
        val_dataset = test_dataset = datasets.__dict__[target[0]](osp.join(root, target[0]), split='test',
                                                                  download=True, transform=val_transform)
        class_names = datasets.MNIST.get_classes()
        num_classes = len(class_names)
    elif dataset_name in datasets.__dict__:
        # load datasets from tllib.vision.datasets
        dataset = datasets.__dict__[dataset_name]

        def concat_dataset(tasks, start_idx, **kwargs):
            # return ConcatDataset([dataset(task=task, **kwargs) for task in tasks])
            return MultipleDomainsDataset([dataset(task=task, **kwargs) for task in tasks], tasks,
                                          domain_ids=list(range(start_idx, start_idx + len(tasks))))

        train_source_dataset = concat_dataset(root=root, tasks=source, download=True, transform=train_source_transform,
                                              start_idx=0)
        train_target_dataset = concat_dataset(root=root, tasks=target, download=True, transform=train_target_transform,
                                              start_idx=len(source))
        val_dataset = concat_dataset(root=root, tasks=target, download=True, transform=val_transform,
                                     start_idx=len(source))
        if dataset_name == 'DomainNet':
            test_dataset = concat_dataset(root=root, tasks=target, split='test', download=True, transform=val_transform,
                                          start_idx=len(source))
        else:
            test_dataset = val_dataset
        class_names = train_source_dataset.datasets[0].classes
        num_classes = len(class_names)
    else:
        raise NotImplementedError(dataset_name)
    return train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, class_names


def validate(val_loader, model, args, device) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    scoremas = AverageMeter('F1 Macro', ':6.2f')
    scoremis = AverageMeter('F1 Micro', ':6.2f')
    precisionmas = AverageMeter('Precision Macro', ':6.2f')
    precisionmis = AverageMeter('Precision Micro', ':6.2f')
    recallmas = AverageMeter('Recall Macro', ':6.2f')
    recallmis = AverageMeter('Recall Micro', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None
    all_targets = []
    all_preds = []
    predict_time = []
    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[:2]
            images = images.to(device)
            target = target.to(device)

            # compute output
            start_time = float(round(time.time() * 1000))
            output = model(images)
            end_time = float(round(time.time() * 1000)) - start_time
            predict_time.append(end_time)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, = accuracy(output, target, topk=(1,))

            if confmat:
                confmat.update(target, output.argmax(1))

            # more measure
            target = target.cpu().numpy()
            output = output.cpu().numpy().argmax(axis=1)
            all_targets += target.tolist()
            all_preds += output.tolist()
            scorema = f1_score(target, output, average="macro")
            precisionma = precision_score(target, output, average="macro")
            scoremi = f1_score(target, output, average="micro")
            precisionmi = precision_score(target, output, average="micro")
            recallmi = recall_score(target, output, average="micro")
            recallma = recall_score(target, output, average="macro")


            # update measure
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            scoremas.update(scorema.item(), images.size(0))
            scoremis.update(scoremi.item(), images.size(0))
            precisionmas.update(precisionma.item(), images.size(0))
            precisionmis.update(precisionmi.item(), images.size(0))
            recallmas.update(recallma.item(), images.size(0))
            recallmis.update(recallmi.item(), images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
        print(" * F1 macro = {:3.5f}".format(scoremas.avg))
        print(" * F1 micro= {:3.5f}".format(scoremis.avg))
        print(" * precision macro= {:3.5f}".format(precisionmas.avg))
        print(" * precision micro= {:3.5f}".format(precisionmis.avg))
        print(" * recall macro = {:3.5f}".format(recallmas.avg))
        print(" * recall micro = {:3.5f}".format(recallmis.avg))
        if confmat:
            print(confmat.format(args.class_names))
    conf_mat = confusion_matrix(all_targets, all_preds)
    report_table = classification_report(all_targets, all_preds, target_names = args.class_names, digits=5, zero_division=0)
    avg_time = float(sum(predict_time)/len(predict_time))
    min_time = min(predict_time)
    max_time = max(predict_time)
    # return top1.avg, losses.avg

    return top1.avg, losses.avg, scoremas.avg, scoremis.avg, precisionmas.avg, precisionmis.avg, recallmas.avg, recallmis.avg, conf_mat, avg_time, min_time, max_time, report_table


def get_train_transform(resizing='default', scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), random_horizontal_flip=True,
                        random_color_jitter=False, resize_size=224, norm_mean=(0.485, 0.456, 0.406),
                        norm_std=(0.229, 0.224, 0.225), auto_augment=None):
    """
    resizing mode:
        - default: resize the image to 256 and take a random resized crop of size 224;
        - cen.crop: resize the image to 256 and take the center crop of size 224;
        - res: resize the image to 224;
    """
    transformed_img_size = 224
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomResizedCrop(224, scale=scale, ratio=ratio)
        ])
    elif resizing == 'cen.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224)
        ])
    elif resizing == 'ran.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomCrop(224)
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
        transformed_img_size = resize_size
    else:
        raise NotImplementedError(resizing)
    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if auto_augment:
        aa_params = dict(
            translate_const=int(transformed_img_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in norm_mean]),
            interpolation=Image.BILINEAR
        )
        if auto_augment.startswith('rand'):
            transforms.append(rand_augment_transform(auto_augment, aa_params))
        else:
            transforms.append(auto_augment_transform(auto_augment, aa_params))
        transforms.append(T.ColorJitter(
            brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
        transforms.append(T.ColorJitter(
            brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])
    return T.Compose(transforms)


def get_val_transform(resizing='default', resize_size=224,
                      norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    """
    resizing mode:
        - default: resize the image to 256 and take the center crop of size 224;
        – res.: resize the image to 224
    """
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
    else:
        raise NotImplementedError(resizing)
    return T.Compose([
        transform,
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])


def empirical_risk_minimization(train_source_iter, model, optimizer, lr_scheduler, epoch, args, device):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)[:2]
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_s, f_s = model(x_s)

        cls_loss = F.cross_entropy(y_s, labels_s)
        loss = cls_loss

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))

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
