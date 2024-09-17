import os
import random
import logging
import argparse
from datetime import datetime

import torch
from torch import optim
from torch.utils.data import DataLoader,random_split, ConcatDataset
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from Dataset_v12 import *
from models.classifier import *
from MD_transform import *
from torchvision import transforms


def define_data_loader(root, batch_size):
    # augment training data
    transform = transforms.Compose([reverse])
    ori_dataset = MD_detection(root=root, transform=None)
    rev_dataset = MD_detection(root=root, transform=transform)

    ori_train, ori_test = random_split(ori_dataset, [int(0.7*len(ori_dataset)), len(ori_dataset)-int(0.7*len(ori_dataset))])
    rev_train, rev_test = random_split(rev_dataset, [int(0.7*len(rev_dataset)), len(rev_dataset)-int(0.7*len(rev_dataset))])

    train = ConcatDataset([ori_train, rev_train])
    test = ori_test

    train_dataloader = DataLoader(train,batch_size=batch_size,shuffle=True)
    test_dataloader = DataLoader(test,batch_size=batch_size)
    return train_dataloader,test_dataloader

def define_logger(args):
    logging_file_name = datetime.now().strftime("%Y-%b-%d-%H-%M")
    logging_file_name = f"{logging_file_name}_{args.model}_{args.transfer_loss}_{args.dis_loss}_{args.experiment_index}.log"
    if not os.path.exists('log'):
        os.makedirs('log')
    log_pth = os.path.join('log',logging_file_name)
    logging.basicConfig(filename=log_pth, level=logging.DEBUG)
    logger = logging.getLogger('myapp')
    logger.info('Start log recording')

    return logger

def define_writer(args):
    formatted_datetime = datetime.now().strftime("%m%d%H%M")
    experiment_name =  f"{formatted_datetime}_{args.model}_{args.transfer_loss}_{args.dis_loss}_{args.experiment_index}"
    writer = SummaryWriter(f"runs/{experiment_name}")
    return writer

def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--model', help='choose a training method', choices=['resnet18','4-conv','10-conv','resnet34'])

    # Dataloader
    parser.add_argument('--lr', help='learning rate',type=float, default=1e-5)
    parser.add_argument('--batch_size', help='batch size',type=int, default=128)

    # Training
    parser.add_argument('--num_epochs', help='number of epochs',type=int, default=500)
    parser.add_argument('--patience', help='patience',type=int, default=300)

    # Domain Adaptation
    parser.add_argument('--transfer_loss_weight',type=float,default=1)
    parser.add_argument('--transfer_loss', type=str, default=None)
    parser.add_argument('--experiment_index',type=int,default=1)

    # Distillation
    parser.add_argument('--dis_loss', type=bool, default=False)
    parser.add_argument('--dis_loss_weight',type=float,default=0.1)
    return parser

def postprocess_args(args):
    i = args.experiment_index
    if i < 4:
        args.source_root = ['Sep12','Sep13']
        if i == 1:
            args.target_root = ['Sep14','Sep15']
            args.generalization_test = ['Sep16']
        if i == 2:
            args.target_root = ['Sep14','Sep16']
            args.generalization_test = ['Sep15']
        if i == 3:
            args.target_root = ['Sep15','Sep16']
            args.generalization_test = ['Sep14']
    elif i > 3 and i < 7:
        args.source_root = ['Sep12','Sep14']
        if i == 4:
            args.target_root = ['Sep13','Sep15']
            args.generalization_test = ['Sep16']
        if i == 5:
            args.target_root = ['Sep13','Sep16']
            args.generalization_test = ['Sep15']
        if i == 6:
            args.target_root = ['Sep15','Sep16']
            args.generalization_test = ['Sep13']
    elif i > 6 and i < 10:
        args.source_root = ['Sep12','Sep15']
        if i == 7:
            args.target_root = ['Sep13','Sep14']
            args.generalization_test = ['Sep16']
        if i == 8:
            args.target_root = ['Sep13','Sep16']
            args.generalization_test = ['Sep14']
        if i == 9:
            args.target_root = ['Sep14','Sep16']
            args.generalization_test = ['Sep13']

    elif i > 9 and i < 13:
        args.source_root = ['Sep12','Sep16']
        if i == 10:
            args.target_root = ['Sep13','Sep14']
            args.generalization_test = ['Sep15']
        if i == 11:
            args.target_root = ['Sep13','Sep15']
            args.generalization_test = ['Sep14']
        if i == 12:
            args.target_root = ['Sep14','Sep15']
            args.generalization_test = ['Sep13']

    elif i > 12 and i < 16:
        args.source_root = ['Sep13','Sep14']
        if i == 13:
            args.target_root = ['Sep12','Sep15']
            args.generalization_test = ['Sep16']
        if i == 14:
            args.target_root = ['Sep12','Sep16']
            args.generalization_test = ['Sep15']
        if i == 15:
            args.target_root = ['Sep15','Sep16']
            args.generalization_test = ['Sep12']
    elif i > 15 and i < 19:
        args.source_root = ['Sep13','Sep15']
        if i == 16:
            args.target_root = ['Sep12','Sep14']
            args.generalization_test = ['Sep16']
        if i == 17:
            args.target_root = ['Sep12','Sep16']
            args.generalization_test = ['Sep14']
        if i == 18:
            args.target_root = ['Sep14','Sep16']
            args.generalization_test = ['Sep12']
    elif i > 18 and i < 22:
        args.source_root = ['Sep13','Sep16']
        if i == 19:
            args.target_root = ['Sep12','Sep14']
            args.generalization_test = ['Sep15']
        if i == 20:
            args.target_root = ['Sep12','Sep15']
            args.generalization_test = ['Sep14']
        if i == 21:
            args.target_root = ['Sep14','Sep15']
            args.generalization_test = ['Sep12']
    elif i > 21 and i < 25:
        args.source_root = ['Sep14','Sep15']
        if i == 22:
            args.target_root = ['Sep12','Sep13']
            args.generalization_test = ['Sep16']
        if i == 23:
            args.target_root = ['Sep12','Sep16']
            args.generalization_test = ['Sep13']
        if i == 24:
            args.target_root = ['Sep13','Sep16']
            args.generalization_test = ['Sep12']
    elif i > 24 and i < 28:
        args.source_root = ['Sep14','Sep16']
        if i == 25:
            args.target_root = ['Sep12','Sep13']
            args.generalization_test = ['Sep15']
        if i == 26:
            args.target_root = ['Sep12','Sep15']
            args.generalization_test = ['Sep13']
        if i == 27:
            args.target_root = ['Sep13','Sep15']
            args.generalization_test = ['Sep12']
    elif i > 27 and i < 31:
        args.source_root = ['Sep15','Sep16']
        if i == 28:
            args.target_root = ['Sep12','Sep13']
            args.generalization_test = ['Sep14']
        if i == 29:
            args.target_root = ['Sep12','Sep14']
            args.generalization_test = ['Sep13']
        if i == 30:
            args.target_root = ['Sep13','Sep14']
            args.generalization_test = ['Sep12']

    args.source_root = [os.path.join(r'G:\Jiarui',f) for f in args.source_root]
    args.target_root = [os.path.join(r'G:\Jiarui',f) for f in args.target_root]
    args.generalization_test_root = [os.path.join(r'G:\Jiarui',f) for f in args.generalization_test]
    return args


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def define_model(args):
    if args.dis_loss:
        model = DisNet(transfer_loss=args.transfer_loss, backbone=args.model)
    else:
        if args.transfer_loss:
            model = TransferNet(transfer_loss=args.transfer_loss, backbone=args.model)
        else:
            model = BaseNet(backbone=args.model)
    return model



def define_opt(model,args):
    opt = optim.Adam(model.parameters(),lr=args.lr)
    return opt


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')