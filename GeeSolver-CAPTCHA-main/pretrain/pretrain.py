import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from functools import partial
import torch.nn as nn
import timm.optim.optim_factory as optim_factory
import time
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import Batch_Balanced_Dataset
from models_mae import MaskedAutoencoderViT
from util import Averager, Logger


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training for Captcha', add_help=False)
    parser.add_argument('--data_path', default="../dataset", help='path to dataset')
    parser.add_argument('--dataset_name', default="google", help='dataset name')
    parser.add_argument('--label_file', default="500.txt", help='file for labeling images')
    parser.add_argument('--unlabeled_number', type=int, default=5000, help='the number of unlabeled images')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_iter', type=int, default=600000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=100, help='Interval between each validation')
    parser.add_argument('--showInterval', type=int, default=5000, help='Interval between Image Shown')
    parser.add_argument('--saveInterval', type=int, default=50000, help='Save Interval between each validation')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--num_layer', type=int, default=8, help='number of encoder layers, decoder is half')
    parser.add_argument('--mask_ratio', type=float, default=0.3, help='mask ratio for reconstruction')
    parser.add_argument('--resume', type=str, default="", help='resume from last training')
    opt = parser.parse_args()

    return opt


def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


if __name__ == '__main__':
    opt = get_args_parser()
    device = torch.device(opt.device)
    print(opt)

    opt.exp_name = f'{opt.dataset_name}-{opt.label_file}-{opt.unlabeled_number}-{opt.num_layer}-{opt.mask_ratio}-{opt.manualSeed}'
    os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)

    sys.stdout = Logger(f'./saved_models/{opt.exp_name}/log.txt', )

    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    datasets = Batch_Balanced_Dataset(opt)

    model = MaskedAutoencoderViT(
        img_size=(70, 200), patch_size=10, embed_dim=512, depth=opt.num_layer, num_heads=4,
        decoder_embed_dim=256, decoder_depth=opt.num_layer // 2, decoder_num_heads=4,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=False)
    model.to(device)

    param_groups = optim_factory.add_weight_decay(model, 0.05)
    optimizer = torch.optim.AdamW(param_groups, lr=1.5e-4, betas=(0.9, 0.95))
    print(optimizer)

    scaler = torch.cuda.amp.GradScaler()

    loss_avg = Averager()
    start_time = time.time()
    iteration = 0

    if opt.resume:
        checkpoint = torch.load(f'./saved_models/{opt.exp_name}/{opt.resume}.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        iteration = checkpoint['iteration']
        scaler.load_state_dict(checkpoint['scaler'])
        print(f"load from {opt.exp_name}/{opt.resume}.pth")

    model.train()
    while True:
        images, _ = datasets.get_batch()
        images = images.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            loss, _, _ = model(images, mask_ratio=opt.mask_ratio)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_avg.add(loss)

        if iteration % opt.valInterval == 0 or iteration == 0:
            elapsed_time = time.time() - start_time
            loss_log = f'[{iteration}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f},' \
                       f' Elapsed_time: {elapsed_time:0.5f}'
            loss_avg.reset()
            print(loss_log)

        if iteration % opt.showInterval == 0 or iteration == 0:
            model.eval()
            with torch.no_grad():
                images_test, _ = datasets.get_batch_test()
                images_test = images_test.to(device)
                loss, y, mask = model(images_test, mask_ratio=opt.mask_ratio)
                y = model.unpatchify(y, img_size=(70, 200))
                y = torch.einsum('nchw->nhwc', y).detach().cpu()

                mask = mask.detach()
                mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
                mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
                mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

                x = torch.einsum('nchw->nhwc', images_test).detach().cpu()

                # masked image
                im_masked = x * (1 - mask)

                # MAE reconstruction pasted with visible patches
                im_paste = x * (1 - mask) + y * mask

                imagenet_mean = np.array([0.5, 0.5, 0.5])
                imagenet_std = np.array([0.5, 0.5, 0.5])

                # make the plt figure larger
                plt.rcParams['figure.figsize'] = [24, 24]

                plt.subplot(1, 4, 1)
                show_image(x[0], "original")

                plt.subplot(1, 4, 2)
                show_image(im_masked[0], "masked")

                plt.subplot(1, 4, 3)
                show_image(y[0], "reconstruction")

                plt.subplot(1, 4, 4)
                show_image(im_paste[0], "reconstruction + visible")

                plt.savefig(f'./saved_models/{opt.exp_name}/{iteration}.png')
            model.train()

        if iteration % opt.saveInterval == 0 or iteration == 0:
            to_save = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': iteration + 1,
                'scaler': scaler.state_dict(),
            }
            torch.save(to_save, f'./saved_models/{opt.exp_name}/{iteration}.pth')

        if iteration == opt.num_iter:
            print('end the training')
            sys.exit()
        iteration += 1
