import os
import threading
import numpy as np
import shutil
from math import log10, exp, pi
from PIL import Image
from datetime import datetime
import logging

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision.models import vgg16
from torchvision.utils import save_image

# --- Perceptual loss network  --- #
class VGG(torch.nn.Module):
    def __init__(self, vgg_model):
        super(VGG, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, dehaze, gt):
        loss = []
        dehaze_features = self.output_features(dehaze)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(F.mse_loss(dehaze_feature, gt_feature))

        return sum(loss)/len(loss)


# --- SSIM network  --- #
class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return (1 - _ssim(img1, img2, window, self.window_size, channel, self.size_average))


# --- Underwater restoration loss network  --- #
class UWLoss(torch.nn.Module):
    """
    Loss for underwater restoration, composed of mse loss, ssim, cosine, vgg(perceptual) loss,
    quantization loss, uicm loss
    """
    def __init__(self, mse=True, ssim=False, angular=True, vgg=False, quant=False, uicm=False):
        super(UWLoss, self).__init__()
        self.mse = mse
        self.ssim = ssim
        self.angular = angular
        self.vgg = vgg
        self.quant = quant
        self.uicm = uicm
        
        self.ssim_module = None
        self.vgg_module = None

        if self.ssim:
            self.ssim_module = SSIM()

        if self.vgg:        
            vgg_model = vgg16(pretrained=True).features[:16]
            vgg_model = vgg_model.cuda()
            for param in vgg_model.parameters():
                param.requires_grad = False
            self.vgg_module = VGG(vgg_model)
        

    def forward(self, img1, img2):
        mse_loss = 0
        ssim_loss = 0
        angular_loss = 0
        vgg_loss = 0
        ce_loss = 0
        uicm_loss = 0

        if self.mse:
            mse_loss = F.smooth_l1_loss(img1, img2)

        if self.ssim:
            ssim_loss = self.ssim_module(img1, img2)

        if self.angular:
            # angular_losses = torch.acos(torch.clamp(F.cosine_similarity(img1, img2), min=-1+1e-12, max=1-1e-12)) / pi
            angular_losses = 1 - F.cosine_similarity(img1, img2)
            angular_loss = torch.mean(angular_losses)  #torch.acos(cos_dist))

        if self.vgg:
            vgg_loss = self.vgg_module(img1, img2)

        if self.quant:
            q_img1, q_img2 = 255.0 * img1, 255.0 * img2
            q_img1, q_img2 = torch.clamp(q_img1.type(torch.long), min=0, max=255), \
                    torch.clamp(q_img2.type(torch.long), min=0, max=255)
            q_img1, q_img2 = q_img1 // 32, q_img2 // 32
            
            oh_img1_r = make_one_hot(q_img1[:, 0, :, :].unsqueeze(1))
            q_img2_r = q_img2[:, 0, :, :].unsqueeze(1)
            oh_img1_g = make_one_hot(q_img1[:, 1, :, :].unsqueeze(1))
            q_img2_g = q_img2[:, 1, :, :].unsqueeze(1)
            oh_img1_b = make_one_hot(q_img1[:, 2, :, :].unsqueeze(1))
            q_img2_b = q_img2[:, 2, :, :].unsqueeze(1)
            
            ce_loss_r = F.cross_entropy(oh_img1_r, q_img2_r.squeeze())
            ce_loss_g = F.cross_entropy(oh_img1_g, q_img2_g.squeeze())
            ce_loss_b = F.cross_entropy(oh_img1_b, q_img2_b.squeeze())
            ce_loss = ce_loss_r + ce_loss_g + ce_loss_b
        
        if self.uicm:
            rg = img1[:, 0, :, :] - img1[:, 1, :, :] # R - G [N, H, W]
            yb = img1[:, 0, :, :]/2 + img1[:, 1, :, :]/2 - img1[:, 2, :, :]
            rg, yb = rg.view(-1), yb.view(-1)
            assert rg.size(0) == yb.size(0)
            idx = rg.size(0) // 10
            rg_mean, rg_std = torch.mean(rg[idx:-idx]), torch.std(rg[idx:-idx])
            yb_mean, yb_std = torch.mean(yb[idx:-idx]), torch.std(yb[idx:-idx])
            uicm_loss = -0.0268*(rg_mean**2 +yb_mean**2)**(0.5) + 0.1568*(rg_std**2 +
                    yb_std**2)**(0.5)

        return mse_loss + ssim_loss + angular_loss + 0.04 * vgg_loss + ce_loss - uicm_loss


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


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), \
            sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight_data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel, sigma = 1.5):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def Gaussiansmoothing(img, channel=3, window_size = 11):
    window = create_window(window_size, channel, sigma=5)
    
    if img.is_cuda:
        window = window.cuda(img.get_device())
    window = window.type_as(img)

    pad = window_size//2
    padded_img = F.pad(img, (pad, pad, pad, pad), mode='reflect')
    x_smooth = F.conv2d(padded_img, window, padding=0, groups=channel)
    
    return x_smooth, img - x_smooth

def psnr(output, target):
    """
    Computes the PSNR.
    1 means the maximum value of intensity(255)
    """
    psnr = 0
    
    with torch.no_grad():
        mse = F.mse_loss(output, target)
        psnr = 10 * log10( 1/ mse.item() )
        
    return psnr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Saves the serialized current checkpoint
    
    Params

    state = 
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(args, optimizer, epoch, prev_lr):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    if args.lr_mode == 'step':
        lr = args.lr * (0.5 ** (epoch // args.step))
    elif args.lr_mode == 'poly':
        lr = args.lr * (1 - epoch / args.epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))
    if lr != prev_lr:
        print('Learning rate has changed!')
        for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    return lr


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def save_output_images(predictions, filenames, output_dir):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # pdb.set_trace()
    for ind in range(len(filenames)):
        # im = Image.fromarray(np.transpose(predictions[ind], (1, 2, 0)).astype(np.uint8)) 
        fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
        out_dir = os.path.split(fn)[0]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        # print(fn)
        pred = predictions[ind]
        save_image(pred, fn)
        # im.save(fn)


def resize_4d_tensor(tensor, width, height):
    tensor_cpu = tensor.cpu().numpy()
    if tensor.size(2) == height and tensor.size(3) == width:
        return tensor_cpu
    out_size = (tensor.size(0), tensor.size(1), height, width)
    out = np.empty(out_size, dtype=np.float32)

    def resize_one(i, j):
        out[i, j] = np.array(
            Image.fromarray(tensor_cpu[i, j]).resize(
                (width, height), Image.BILINEAR))

    def resize_channel(j):
        for i in range(tensor.size(0)):
            out[i, j] = np.array(
                Image.fromarray(tensor_cpu[i, j]).resize(
                    (width, height), Image.BILasdfINEAR))

    # workers = [threading.Thread(target=resize_one, args=(i, j))
    #            for i in range(tensor.size(0)) for j in range(tensor.size(1))]

    workers = [threading.Thread(target=resize_channel, args=(j,))
               for j in range(tensor.size(1))]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    # for i in range(tensor.size(0)):
    #     for j in range(tensor.size(1)):
    #         out[i, j] = np.array(
    #             Image.fromarray(tensor_cpu[i, j]).resize(
    #                 (w, h), Image.BILINEAR))
    # out = tensor.new().resize_(*out.shape).copy_(torch.from_numpy(out))
    return out


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


# def create_window(window_size, channel):
#     _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
#     _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#     window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
#     return window


def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def make_one_hot(labels, class_num=8):
    one_hot = torch.cuda.FloatTensor(labels.size(0), class_num, labels.size(2), labels.size(3))
    target = one_hot.scatter_(1, labels.data, 1)
    return Variable(target)


def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

