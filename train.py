import os
import time
import shutil
import sys
import logging
import itertools
from datetime import datetime
from network import generator
from dataset import DehazeList
from utils import adjust_learning_rate, save_output_images, save_checkpoint, psnr,\
    AverageMeter
    
import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import transforms
import data_transforms as transforms
from torch.autograd import Variable

def train(train_loader, model, criterion, optimizer, epoch, eval_score=None, \
    print_freq=10, logger=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()

    # switch to train mode
    net = model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        real_hazy = input.float()
        real_clear = target.float()

        real_hazy = real_hazy.cuda()
        real_clear = real_clear.cuda(async=True)

        real_hazy = torch.autograd.Variable(real_hazy)
        real_clear = torch.autograd.Variable(real_clear)
        
        ##### Generator #####
        optimizer.zero_grad()

        # Loss
        fake_clear, _, _ = net(real_hazy)
        loss = criterion(fake_clear, real_clear)

        loss.backward()

        optimizer.step()

        losses.update(loss.data, input.size(0))

        ###########################

        # measure psnr
        if eval_score is not None:
            scores.update(eval_score(fake_clear, real_clear), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == (print_freq-1):
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Score {top1.val:.3f} ({top1.avg:.3f})'.format(
                (epoch+1), (i+1), len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=scores))


def validate(val_loader, model, criterion, print_freq=10, output_dir='val', \
    save_vis=False, epoch=None, eval_score=None, logger=None, auto_save=True, best_score=0.0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    score = AverageMeter()

    # switch to evaluate mode
    generator = model
    generator.eval()

    end = time.time()
    for i, (input, target, name) in enumerate(val_loader):
        input = input.float()
        target = target.float()

        input = input.cuda()
        target = target.cuda(async=True)

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        with torch.no_grad():
            output, _, _ = generator(input_var)

        loss = criterion(output, target_var)
        losses.update(loss.data, input.size(0))

        if eval_score is not None:
            score.update(eval_score(output, target_var), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        save_dir = os.path.join(output_dir, 'epoch_{:04d}'.format(epoch+1))
        
        if save_vis == True:
            save_dir = os.path.join(output_dir, 'epoch_{:04d}'.format(epoch+1))
            pred = output
            save_output_images(pred, name, save_dir)

        if auto_save == True and (score.avg) > best_score :
            save_dir = os.path.join(output_dir, 'best/'+'epoch_{:04d}'.format(epoch+1))
            pred = output
            save_output_images(pred, name, save_dir)
            logger.info('Best model: {0}'.format(epoch+1))
        
        if i % print_freq == 0:
            logger.info('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Score {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time,
                loss=losses, top1=score))

    logger.info(' * Score {top1.avg:.3f}'.format(top1=score))
    print()

    return score.avg


def train_dehaze(args, saveDirName='.', logger=None):
    batch_size = args.batch_size
    num_workers = args.workers
    crop_size = args.crop_size
    
    
    print(' '.join(sys.argv))

    # logging hyper-parameters
    for k, v in args.__dict__.items():
        logger.info('{0}:\t{1}'.format(k, v))
    
    # Generators
    net = generator(3, 3)
    net = nn.DataParallel(net).cuda()
    
    model = net
        
    # Criterion for updating weights
    criterion = nn.L1Loss()
    criterion = criterion.cuda()

    # Data loading code
    data_dir = args.data_dir

    t = []
    if args.random_scale > 0:
        t.append(transforms.RandomScale(args.random_scale))
    t.append(transforms.RandomCrop(crop_size))
    if args.random_rotate > 0:
        t.append(transforms.RandomRotate(args.random_rotate))
    t.extend([transforms.RandomHorizontalFlip(),
              transforms.RandomVerticalFlip(),
            #   transforms.AddNoise(),
              transforms.RandomIdentityMapping(p=0.4),
              transforms.ToTensor(),
    ])


    # DataLoaders for training/validation dataset
    train_loader = torch.utils.data.DataLoader(
        DehazeList(data_dir, 'train', transforms.Compose(t),
                list_dir=args.list_dir, out_name=False),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        DehazeList(data_dir, 'val', transforms.Compose([
            transforms.ToTensor(),
        ]), list_dir=args.list_dir, out_name=True),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=False, drop_last=False
    )

    # define loss function (criterion) and optimizer
    
    
    optimizer = torch.optim.Adam(net.parameters(),
                                args.lr,
                                betas=(0.5, 0.999),
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True
    best_psnr1 = 0
    start_epoch = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_psnr1 = checkpoint['best_psnr1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    if args.evaluate:
        validate(val_loader, model, criterion, eval_score=psnr, logger=logger)
        return

    lr = args.lr
    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(args, optimizer, epoch, lr)
        logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch+1, lr))

        train(train_loader, model, criterion, optimizer, epoch, eval_score=psnr, logger=logger)

        psnr1 = 0
        
        if epoch % 5 == 4:
            psnr1 = validate(val_loader, model, criterion, eval_score=psnr, save_vis=True, \
                                epoch=epoch, logger=logger, best_score=best_psnr1)
        else:
            psnr1 = validate(val_loader, model, criterion, eval_score=psnr, epoch=epoch, \
                                logger=logger, best_score=best_psnr1)
        
        if epoch == 0:
            best_psnr1 = psnr1

        is_best = psnr1 >= best_psnr1
        best_psnr1 = max(psnr1, best_psnr1)
        
        checkpoint_path = saveDirName + '/'  + 'checkpoint_latest.pth.tar'

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_psnr1': best_psnr1,
        }, is_best, filename=checkpoint_path)

        if (epoch + 1) % 1 == 0:
            history_path = saveDirName + '/' + 'checkpoint_{:03d}.pth.tar'.format(epoch + 1)
            shutil.copyfile(checkpoint_path, history_path)
