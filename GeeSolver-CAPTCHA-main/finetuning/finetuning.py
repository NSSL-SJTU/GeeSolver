import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from functools import partial
import torch.nn as nn
import time
import sys
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from test import validation
from dataset import Batch_Balanced_Dataset
from models_mae import VisionTransformer, TransformerDecoder
from util import AttnLabelConverter, Averager, Logger


def get_args_parser():
    parser = argparse.ArgumentParser('MAE finetuning for Captcha', add_help=False)
    parser.add_argument('--data_path', default="../dataset", help='path to dataset')
    parser.add_argument('--dataset_name', default="google", help='dataset name')
    parser.add_argument('--label_file', default="500.txt", help='file for labeling images')
    parser.add_argument('--unlabeled_number', type=int, default=5000, help='the number of unlabeled images')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_iter', type=int, default=100000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=500, help='Interval between each validation')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--batch_max_length', type=int, default=10, help='maximum-label-length')
    parser.add_argument('--num_layer', type=int, default=8, help='number of encoder layers')
    parser.add_argument('--mask_ratio', type=float, default=0.3, help='mask ratio for reconstruction')
    parser.add_argument('--restore', type=int, default=0, help='number of iters to restore')
    parser.add_argument('--cut', action="store_false",
                        help='whether to cut when finetuning(only for Google; captchas default: True)')
    parser.add_argument('--compression', type=int, default=1,
                        help='1 for up and down; 2 for cross; 3 for one')
    parser.add_argument('--num', type=int, default=-1,
                        help='how many columns to pay attention to (default: -1 for all)')
    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    opt = get_args_parser()
    device = torch.device(opt.device)
    print(opt)

    opt.exp_name = f'{opt.dataset_name}-{opt.label_file}-{opt.unlabeled_number}-{opt.num_layer}-{opt.mask_ratio}-{opt.manualSeed}'
    os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)

    sys.stdout = Logger(f'./saved_models/{opt.exp_name}/{opt.restore}_{opt.compression}_{opt.cut}_{opt.num}_log.txt', )

    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    datasets = Batch_Balanced_Dataset(opt)

    encoder = VisionTransformer(
        img_size=(70, 200), patch_size=10, embed_dim=512, depth=opt.num_layer, num_heads=4, qkv_bias=True,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), num=opt.num)
    encoder.to(device)
    decoder = TransformerDecoder(
        len(converter.character), opt.batch_max_length+2, opt.compression
    )
    decoder.to(device)

    decoder_ema = TransformerDecoder(
        len(converter.character), opt.batch_max_length + 2, opt.compression
    )
    decoder_ema.to(device)

    for param_main, param_ema in zip(decoder.parameters(), decoder_ema.parameters()):
        param_ema.data.copy_(param_main.data)  # initialize
        param_ema.requires_grad = False  # not update by gradient

    checkpoint = torch.load(f'../pretrain/saved_models/{opt.exp_name}/{opt.restore}.pth')
    msg = encoder.load_state_dict(checkpoint['model'], strict=False)
    print(msg)

    params = list(filter(lambda p: p.requires_grad, decoder.parameters()))
    optimizer = optim.SGD(params, lr=0.02, momentum=0.9, weight_decay=5e-4, nesterov=True)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    loss_1_avg = Averager()
    loss_2_avg = Averager()
    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    iteration = 0
    acc_list = []

    scaler = torch.cuda.amp.GradScaler()

    encoder.eval()
    decoder.train()
    decoder_ema.train()

    while True:
        image, text, image_w, image_s = datasets.get_batch()
        # images, text = datasets.get_batch()
        image = image.to(device)
        image_w = image_w.to(device)
        image_s = image_s.to(device)
        text = converter.encode(text, batch_max_length=opt.batch_max_length)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                out = encoder(torch.cat([image, image_w, image_s], dim=0)).detach()
            encoder_out = out[:opt.batch_size]
            image_w, image_s = out[opt.batch_size:].chunk(2)
            decoder_out = decoder.forward_train(encoder_out, text)
            target = text[:, 1:]
            loss_1 = criterion(decoder_out.view(-1, decoder_out.shape[-1]), target.contiguous().view(-1))

            logits_u_w = decoder_ema.forward_test(image_w).detach()
            logits_u_s = decoder.forward_test(image_s)

            logits_u_w = logits_u_w.view(-1, logits_u_w.shape[-1])
            logits_u_s = logits_u_s.view(-1, logits_u_s.shape[-1])
            pseudo_label = torch.softmax(logits_u_w, dim=-1)

            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(0.95).float()
            loss_2 = 2 * (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()

            loss = loss_1 + loss_2

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 10)
        scaler.step(optimizer)
        scaler.update()

        for ema_param, param in zip(decoder_ema.parameters(), decoder.parameters()):
            ema_param.data.mul_(0.999).add_(1 - 0.999, param.data)

        loss_1_avg.add(loss_1)
        loss_2_avg.add(loss_2)

        if (iteration + 1) % opt.valInterval == 0 or iteration == 0:
            elapsed_time = time.time() - start_time
            decoder.eval()
            decoder_ema.eval()
            with torch.no_grad():
                valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data = validation(
                    encoder, decoder_ema, criterion, datasets.test_data_loader, converter, opt)
            decoder.train()
            decoder_ema.train()

            loss_log = f'[{iteration + 1}/{opt.num_iter}] Label loss: {loss_1_avg.val():0.5f}, Unlabel loss: {loss_2_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
            loss_1_avg.reset()
            loss_2_avg.reset()

            acc_list.append(current_accuracy)
            current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.2f}'

            # keep best accuracy model (on valid dataset)
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
            if current_norm_ED > best_norm_ED:
                best_norm_ED = current_norm_ED
            best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}'

            loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
            print(loss_model_log)

            # show some predicted results
            dashed_line = '-' * 80
            head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
            predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
            for gt, pred, confidence in zip(labels[:5], preds[:5], confidence_score[:5]):
                gt = gt[:gt.find('[s]')]
                pred = pred[:pred.find('[s]')]

                predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
            predicted_result_log += f'{dashed_line}'
            print(predicted_result_log)

            if len(acc_list) % 20 == 0 and len(acc_list) > 0:
                fig = plt.figure(figsize=(20, 10))
                ax = fig.add_subplot(111)
                ax.plot(acc_list)
                test_acc_array = np.array(acc_list)
                max_indx = np.argmax(test_acc_array)
                show_max = '[' + str(max_indx) + " " + str(test_acc_array[max_indx].item()) + ']'
                ax.annotate(show_max, xytext=(max_indx, test_acc_array[max_indx].item()),
                             xy=(max_indx, test_acc_array[max_indx].item()))
                fig.savefig(f'./saved_models/{opt.exp_name}/{opt.restore}_{opt.compression}_{opt.cut}_{opt.num}.png')

        if (iteration + 1) == opt.num_iter:
            print('end the training')
            sys.exit()
        iteration += 1
