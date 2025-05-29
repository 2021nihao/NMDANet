import os
import shutil
import json
import time

from apex import amp
from torch.cuda import amp

import numpy as np

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from toolbox import get_dataset
from toolbox import get_logger
from toolbox import get_model
from toolbox import averageMeter, runningScore
from toolbox import save_ckpt
from toolbox import Ranger
from toolbox import setup_seed
from toolbox.losses import pytorch_iou
from toolbox.losses.edge_loss import EdgeHoldLoss
from toolbox.losses.pytorch_ssim import SSIM
from toolbox.losses.Dice_loss import BinaryDiceLoss
from toolbox.losses.focal_loss import BinaryFocalLoss
from toolbox.losses.loss import dice_loss
from toolbox import LovaszSoftmax
from toolbox.models.NMCFNet_b4_dc import NMCFNet

setup_seed(33)



class BCELOSS(nn.Module):
    def __init__(self):
        super(BCELOSS, self).__init__()
        self.nll_lose = nn.BCELoss()

    def forward(self, input_scale, taeget_scale):
        losses = []
        for inputs, targets in zip(input_scale, taeget_scale):
            lossall = self.nll_lose(inputs, targets)
            losses.append(lossall)
        total_loss = sum(losses)
        return total_loss

class eeemodelLoss(nn.Module):

    def __init__(self, class_weight=None, ignore_index=-100, reduction='mean'):
        super(eeemodelLoss, self).__init__()

        self.class_weight_semantic = torch.from_numpy(np.array(
                [1.5105, 16.6591, 29.4238, 34.6315, 40.0845, 41.4357, 47.9794, 45.3725, 44.9000])).float()
        self.class_weight_binary = torch.from_numpy(np.array([1.5121, 10.2388])).float()
        self.class_weight_boundary = torch.from_numpy(np.array([1.4459, 23.7228])).float()

        self.class_weight = class_weight
        self.LovaszSoftmax = LovaszSoftmax()
        self.MSELoss = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.semantic_loss = nn.CrossEntropyLoss(weight=self.class_weight_semantic)
        # self.semantic_loss = nn.CrossEntropyLoss()
        self.binary_loss = nn.CrossEntropyLoss(weight=self.class_weight_binary)
        self.boundary_loss = nn.CrossEntropyLoss(weight=self.class_weight_boundary)

        self.IOU = pytorch_iou.IOU(size_average = True).cuda()
        self.BCE = nn.BCEWithLogitsLoss()
        # self.BCE_w = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.56))
        self.KLD = nn.KLDivLoss(reduction='mean', log_target=True)
        self.edge_with_logits = EdgeHoldLoss()
        self.Dice_loss = BinaryDiceLoss()
        self.dice_loss = dice_loss
        self.focal_binary = BinaryFocalLoss()
        self.ssim = SSIM(size_average=False)
        # self.dice_loss = DiceLoss()

    def forward(self, s_logits, t_logits, label ):
        t3, t2, t1, t_r, t_t = t_logits
        s3, s2, s1, s_r, s_t = s_logits
        B, _, _, _ = t_logits[0].shape

        t_r_s, t_t_s, t3_s = torch.sigmoid(t_r/5.0), torch.sigmoid(t_t/5.0), torch.sigmoid(t3/5.0)
        s_r_s, s_t_s, s1_s, s2_s, s3_s = s_r/5.0, s_t/5.0, s1/5.0, s2/5.0, s3/5.0

        label = torch.unsqueeze(label, 1).float()

        b1, b2, b3, b4, b5 = 0, 0, 0, 0, 0
        loss_r_s, loss_t_s, loss_t1, loss_t2, loss_t3 = 0, 0, 0, 0, 0
        for i in range(B):
        # for i, si in zip(range(B), [s1_s, s2_s, s3_s]):
            score_t = self.ssim(t3[i:i + 1, :, :, :], label[i:i + 1, :, :, :])
            score_s = self.ssim(s3[i:i + 1, :, :, :], label[i:i + 1, :, :, :])
            score_each = (score_t + 3*score_s) / 4

            s_r_s_detach, s_t_s_detach, s1_s_detach, s2_s_detach, s3_s_detach = (s_r_s[i:i + 1, :, :, :], s_t_s[i:i + 1, :, :, :], s1_s[i:i + 1, :, :, :], s2_s[i:i + 1, :, :, :], s3_s[i:i + 1, :, :, :])
            t_r_s_detach, t_t_s_detach, t3_s_detach = (t_r_s[i:i + 1, :, :, :], t_t_s[i:i + 1, :, :, :], t3_s[i:i + 1, :, :, :])

            rate = math.exp(1-score_each)

            loss_r_each = rate * (self.KLD((torch.sigmoid(s_r_s_detach)).log(), t_t_s_detach.log()) + self.dice_loss(torch.sigmoid(s_r_s_detach), t_t_s_detach))
            loss_t_each = rate * (self.KLD((torch.sigmoid(s_t_s_detach)).log(), t_r_s_detach.log()) + self.dice_loss(torch.sigmoid(s_t_s_detach), t_r_s_detach))

            loss_t1_each = rate * (self.KLD((torch.sigmoid(s1_s_detach)).log(), t3_s_detach.log()) + self.dice_loss(torch.sigmoid(s1_s_detach), t3_s_detach))
            loss_t2_each = rate * (self.KLD((torch.sigmoid(s2_s_detach)).log(), t3_s_detach.log()) + self.dice_loss(torch.sigmoid(s2_s_detach), t3_s_detach))
            loss_t3_each = rate * (self.KLD((torch.sigmoid(s3_s_detach)).log(), t3_s_detach.log()) + self.dice_loss(torch.sigmoid(s3_s_detach), t3_s_detach))

            if loss_r_each == loss_r_each and loss_r_each != float("inf"):
                loss_r_s += loss_r_each
                b1 += 1
            if loss_t_each == loss_t_each and loss_t_each != float("inf"):
                loss_t_s += loss_t_each
                b2 += 1
            if loss_t1_each == loss_t1_each and loss_t1_each != float("inf"):
                loss_t1 += loss_t1_each
                b3 += 1
            if loss_t2_each == loss_t2_each and loss_t2_each != float("inf"):
                loss_t2 += loss_t2_each
                b4 += 1
            if loss_t3_each == loss_t3_each and loss_t3_each != float("inf"):
                loss_t3 += loss_t3_each
                b5 += 1
            print("each:", loss_r_each, loss_t_each, loss_t1_each, loss_t2_each, loss_t3_each)
        # ###软监督
        if b1!=0:
            loss_r_s /= b1
        if b2 != 0:
            loss_t_s /= b2
        if b3 != 0:
            loss_t1 /= b3
        if b4 != 0:
            loss_t2 /= b4
        if b5 != 0:
            loss_t3 /= b5

        ##硬监督
        loss_r_h = self.IOU(torch.sigmoid(s_r), label) + self.BCE(s_r, label)
        loss_t_h = self.IOU(torch.sigmoid(s_t), label) + self.BCE(s_t, label)

        loss_1 = self.IOU(torch.sigmoid(s1), label) + self.BCE(s1, label)
        loss_2 = self.IOU(torch.sigmoid(s2), label) + self.BCE(s2, label)
        loss_3 = self.IOU(torch.sigmoid(s3), label) + self.BCE(s3, label)

        print("Y:", loss_1, loss_2, loss_3, loss_r_h, loss_t_h, "\n", "s:", loss_t1, loss_t2, loss_t3, loss_r_s, loss_t_s)

        loss = 3*loss_1 + 3*loss_2 + 4*loss_3 + 3*loss_r_h + 3*loss_t_h + 2*loss_t1 + 2*loss_t2 + 4*loss_t3 + 2*loss_r_s + 2*loss_t_s
        # loss = loss_t1 + loss_t2 + 2*loss_t3
        torch.cuda.empty_cache()

        return loss


def run(args):
    torch.cuda.set_device(args.cuda)
    with open(args.config, 'r') as fp:
        cfg = json.load(fp)

    logdir = f'run/{time.strftime("%Y-%m-%d-%H-%M")}({cfg["dataset"]}-{cfg["model_name"]})/'
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info(f'Conf | use logdir {logdir}')

    # model
    S_model = get_model(cfg)
    device = torch.device(f'cuda:{args.cuda}')
    S_model.to(device)

    # T_model = restart().to(device)
    # T_Weight = "/media/user/shuju/zh/CVPR2021_PDNet-main/run/2022-11-24-15-40(mirrorrgbd-new_year_convnext_128_5)/132model.pth"
    T_model = NMCFNet().to(device)
    # T_Weight = "/media/user/shuju/zh/Mirror_xtx/run/2023-03-01-11-19(mirrorrgbd_xtx-NMCFNet_b3)/168model.pth"
    T_Weight = "/media/user/shuju/zh/Mirror_xtx/run/2023-04-03-22-34(mirrorrgbd_xtx-NMCFNet_b4_dc)/124model.pth"
    T_model.load_state_dict(torch.load(T_Weight, map_location={'cuda:2': 'cuda:2'}), strict=False)
    for p in T_model.parameters():
        p.stop_gradient = True
    T_model.eval()
    # T_train_logits = teacher_predict(model=T_model, loader=train_loader, inputs="rgbd", loca=T_Weight)
    total_params = sum(p.numel() for p in S_model.parameters())
    print("S_model have " + f'{total_params:,} total parameters.')

    trainset, testset = get_dataset(cfg)
    train_loader = DataLoader(trainset, batch_size=cfg['ims_per_gpu'], shuffle=True, num_workers=cfg['num_workers'],
                              pin_memory=True)

    test_loader = DataLoader(testset, batch_size=cfg['ims_per_gpu'], shuffle=False, num_workers=cfg['num_workers'],
                              pin_memory=True)

    params_list = S_model.parameters()
    optimizer = Ranger(params_list, lr=cfg['lr_start'], weight_decay=cfg['weight_decay'])
    scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep / cfg['epochs']) ** 0.9)
    Scaler = amp.GradScaler()
    # criterion = LovaszSoftmax().to(device)
    train_criterion = eeemodelLoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    # 指标 包含unlabel
    train_loss_meter = averageMeter()
    test_loss_meter = averageMeter()
    running_metrics_test = runningScore(cfg['n_classes'], ignore_index=cfg['id_unlabel'])

    best_test = 100

    # 每个epoch迭代循环
    for ep in range(cfg['epochs']):

        # training
        S_model.train()
        T_model.eval()
        train_loss_meter.reset()
        for i, sample in enumerate(train_loader):
            optimizer.zero_grad()  # 梯度清零

            ################### train edit #######################
            if cfg['inputs'] == 'rgb':
                image = sample['image'].to(device)
                label = sample['label'].to(device)
                # bound = sample['bound'].to(device)
                # binary_label = sample['binary_label'].to(device)
                # targets = [label, binary_label, bound]
                # print(label.shape, bound.shape)
                # targets = [label, bound]
                predict = S_model(image)
                predict_T = T_model(image)
                # print(image.shape, label.shape, predict[0].shape,"111")
                # print(predict[0].shape)
            else:
                image = sample['image'].to(device)
                depth = sample['depth'].to(device)
                # depth_normalized = sample['depth_normalized'].to(device)
                label = sample['label'].to(device)

            with amp.autocast():
                predict = S_model(image, depth)
                predict_T = T_model(image, depth)
                loss = train_criterion( predict, predict_T, label)
            ####################################################

            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            # loss.backward()
            # optimizer.step()
            Scaler.scale(loss).backward()
            Scaler.step(optimizer)
            Scaler.update()

            train_loss_meter.update(loss.item())

        scheduler.step(ep)


        # test
        with torch.no_grad():
            S_model.eval()
            running_metrics_test.reset()
            # misc.compute_ber.reset()
            difficult = []
            avg_iou, img_num1 = 0.0, 0.0
            avg_ber, img_num2 = 0.0, 0.0
            avg_mae, img_num3 = 0.0, 0.0
            test_loss_meter.reset()
            for i, sample in enumerate(test_loader):
                if cfg['inputs'] == 'rgb':
                    image = sample['image'].to(device)
                    label = sample['label'].to(device)
                    predict = S_model(image)[0]
                    # print(image.shape, label.shape, predict.shape,"222")
                else:
                    image = sample['image'].to(device)
                    depth = sample['depth'].to(device)
                    label = sample['label'].to(device)

                    predict = S_model(image, depth)[0]

                label = torch.unsqueeze(label, 1).float()
                IOU = pytorch_iou.IOU()
                edge_loss = EdgeHoldLoss()
                loss = IOU(torch.sigmoid(predict), label) + F.binary_cross_entropy_with_logits(predict, label) + edge_loss(predict, label)
                test_loss_meter.update(loss.item())

                gt1 = gt2 = gt3 = label.cuda()
                pre1 = pre2 = pre3 = torch.sigmoid(predict).cuda()

                #iou
                pre1 = (pre1 >= 0.5)
                gt1 = (gt1 >= 0.5)
                iou = torch.sum((pre1 & gt1)) / torch.sum((pre1 | gt1))
                if iou == iou:  # for Nan
                    avg_iou += iou
                    img_num1 += 1.0

                #ber
                pre2 = (pre2 >= 0.5)
                gt2 = (gt2 >= 0.5)
                N_p = torch.sum(gt2) + 1e-20
                N_n = torch.sum(torch.logical_not(gt2)) + 1e-20  # should we add this？
                TP = torch.sum(pre2 & gt2)
                TN = torch.sum(torch.logical_not(pre2) & torch.logical_not(gt2))
                ber = 1 - (1 / 2) * ((TP / N_p) + (TN / N_n))

                if ber == ber:  # for Nan
                    avg_ber += ber
                    img_num2 += 1.0

                #mae
                pre3 = torch.where(pre3 >= 0.5, torch.ones_like(pre3), torch.zeros_like(pre3))
                gt3 = torch.where(gt3 >= 0.5, torch.ones_like(gt3), torch.zeros_like(gt3))
                mea = torch.abs(pre3 - gt3).mean()

                if mea == mea:  # for Nan
                    avg_mae += mea
                    img_num3 += 1.0
        #iou结果
        avg_iou /= img_num1
        test_iou = avg_iou.item()
        # print(avg_iou.item())

        # ber结果
        avg_ber /= img_num2
        test_ber = avg_ber.item() * 100
        # print(avg_ber.item() * 100)

        # mae结果
        avg_mae /= img_num3
        test_mae = avg_mae.item()

        train_loss = train_loss_meter.avg
        test_loss = test_loss_meter.avg

        test_avg = (test_ber + test_mae) / 2

        logger.info(
        f'Iter | [{ep + 1:3d}/{cfg["epochs"]}] loss={train_loss:.3f}/{test_loss:.3f}, iou={test_iou:.5f}, ber={test_ber:.5f}, mae={test_mae:.5f}')
        if test_avg < best_test and test_iou >= 0.500 and test_ber <= 16.50 and test_mae <= 0.119:
            best_test = test_avg
            save_ckpt(logdir, S_model)
        if ep >= 0.95 * cfg['epochs']:
            name = f'{ep + 1}' + "_"
            save_ckpt(logdir, S_model, name)

        if test_iou >= 0.805 and test_ber <= 7.55000 and test_mae <= 0.0360:
            # logdir1 = logdir+"{}".format(ep)
            name = f'{ep+1}'
            save_ckpt(logdir, S_model, name)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", type=str, default="/XX/XX/XXX/zh/XXX/configs/mirror_KDDO.json", help="Configuration file to use")
    parser.add_argument("--opt_level", type=str, default='O1')
    parser.add_argument("--inputs", type=str.lower, default='rgb', choices=['rgb', 'rgbd'])
    parser.add_argument("--resume", type=str, default='',
                        help="use this file to load last checkpoint for continuing training")
    parser.add_argument("--cuda", type=int, default=2, help="set cuda device id")
    parser.add_argument("--备注", type=str, default="", help="记录配置和对照组")

    args = parser.parse_args()

    run(args)
