import argparse
import logging
import os
import threading
import time
import numpy as np
import shutil
import itertools
from os.path import join, exists, split
from math import log10
from train import train_dehaze
from test import test_dehaze

import sys
from PIL import Image
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from datetime import datetime
from dataset import DehazeList
import data_transforms as transforms


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('cmd', choices=['train', 'test']) #
    parser.add_argument('-d', '--data-dir', default=None, required=True) #
    parser.add_argument('-s', '--crop-size', default=0, type=int) #
    parser.add_argument('--step', type=int, default=200) #
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', #
                        help='input batch size for training (default: 64)') #
    parser.add_argument('--epochs', type=int, default=10, metavar='N', #
                        help='number of epochs to train (default: 10)') #
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)') #
    parser.add_argument('--lr-mode', type=str, default='step') #
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, #
                        metavar='W', help='weight decay (default: 1e-4)') #
    parser.add_argument('--resume', default='', type=str, metavar='PATH', #
                        help='path to latest checkpoint of model (default: none)') #
    parser.add_argument('-j', '--workers', type=int, default=8) #
    parser.add_argument('--phase', default='val') #
    parser.add_argument('--random-scale', default=0, type=float)
    parser.add_argument('--random-rotate', default=0, type=int)
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    
    return args


def make_data_txt(data_dir, list_dir = './datasets'):
    list_dir = list_dir
    if not os.path.exists(list_dir):
        os.makedirs(list_dir, exist_ok=True)
    
    phase_list = ['train', 'val', 'test']
    img_type_list = list(zip(['image', 'gt'], ['HAZY', 'GT']))

    for phase, img_type in itertools.product(phase_list, img_type_list):
        # print(phase)
        dir = os.path.join(data_dir, phase, img_type[1])
        if os.path.exists(dir):
            f = open(os.path.join(list_dir, phase + '_' + img_type[0] + '.txt'), 'w')
            
            img_list = [os.path.join(img_type[1], img) \
                            for img in os.listdir(dir) if (img.endswith('png') or img.endswith('jpg'))]
            img_list.sort()
            
            for i in range(16):
                for item in img_list:
                    f.write(item + '\n')
                if (i == 0) & (phase != 'train'):
                    break

            f.close()
    
    
def main():
    args = parse_args()

    make_data_txt(data_dir=args.data_dir)
    
    dt_now = datetime.now()
    timeName = "{:4d}{:02d}{:02d}{:02d}{:02d}".format(dt_now.year, dt_now.month, \
    dt_now.day, dt_now.hour, dt_now.minute)
    saveDirName = './runs/train/' + timeName
    if not os.path.exists(saveDirName):
        os.makedirs(saveDirName, exist_ok=True)

    # logging configuration
    FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(saveDirName + '/log_training.log')
    logger.addHandler(file_handler)

    if args.cmd == 'train':
        train_dehaze(args, saveDirName=saveDirName, logger=logger)
    if args.cmd == 'test':
        test_dehaze(args, logger=logger)

if __name__ == '__main__':
    main()
