import os
import time
from network import generator
from dataset import DehazeList

from utils import save_output_images, AverageMeter, Gaussiansmoothing

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import data_transforms as transforms

def test(eval_data_loader, model, output_dir='test', save_vis=True, logger=None):
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    for iter, (image, name) in enumerate(eval_data_loader):
        data_time.update(time.time() - end)
        image_var = Variable(image)
        _, _, h, w = image_var.size()

        if (h%8, w%8) != (0, 0):
            image_var = F.interpolate(image_var, size=(h + 8 - h % 8 , w + 8 - w % 8),
                    mode='bilinear')
        
        with torch.no_grad():
            final, trans, light = model(image_var)

        pred = final
        batch_time.update(time.time() - end)

        if save_vis:
            save_output_images(pred, name, output_dir)
            save_output_images(trans, name, output_dir+'transmission')
            save_output_images(light, name, output_dir+'light')

        end = time.time()
        logger.info('Eval: [{0:04d}/{1:04d}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(iter+1, len(eval_data_loader), batch_time=batch_time,
                            data_time=data_time))

def test_dehaze(args, logger=None):
    batch_size = args.batch_size
    num_workers = args.workers
    phase = args.phase

    for k, v in args.__dict__.items():
        print(k, ':', v)

    model = generator(3, 3)
    model = nn.DataParallel(model).cuda()
    
    data_dir = args.data_dir

    dataset = DehazeList(data_dir, phase, transforms.Compose([
        transforms.ToTensor(),
        #normalize,
        ]), list_dir=args.list_dir, out_name=True)

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=False
    )

    cudnn.benchmark = True

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_psnr1 = checkpoint['best_psnr1']
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))
            return


    out_dir = '{:03d}_{}'.format(start_epoch, phase)
    test(test_loader, model, save_vis=True, output_dir=out_dir, logger=logger)
