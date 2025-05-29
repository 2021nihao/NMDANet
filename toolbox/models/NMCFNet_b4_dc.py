import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, trunc_normal_
from torch.nn.parameter import Parameter
from torch.nn import init

from backbone.SegFormer_master.mmseg.models.backbones.mix_transformer import mit_b3, mit_b0, mit_b4


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, groups=1, bn=True, relu=True):
        padding = ((kernel_size - 1) * dilation + 1) // 2
        # padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                              bias=False if bn else True)
        self.bn = bn
        if bn:
            self.bnop = nn.BatchNorm2d(out_planes)
            # self.bnop = nn.InstanceNorm2d(out_planes)
        self.relu = relu
        if relu:
            # self.reluop = nn.ReLU6(inplace=True)
            self.reluop = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bnop(x)
        if self.relu:
            x = self.reluop(x)
        return x

class NM(nn.Module):
    def __init__(self, c1, c2, c):
        super(NM, self).__init__()
        self.conv1_f1 = nn.Conv2d(c1, c, 1, 1)
        self.conv1_f2 = nn.Conv2d(c2, c, 1, 1)
        self.conv_f2_msak = nn.Conv2d(c, 1, 1, 1)
        self.upx2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.cbl_add = ConvBNReLU(in_planes=c, out_planes=c, kernel_size=3, stride=1)
        self.cbl_mul = ConvBNReLU(in_planes=c, out_planes=c, kernel_size=3, stride=1)
        self.cbl_cat = ConvBNReLU(in_planes=3*c, out_planes=c, kernel_size=3, stride=1)
        self.extract_edges = Extract_edges()

    def forward(self, f1, f2):
        # B, C, H, W = f1.shape
        # print(f1.shape, f2.shape)
        # mask_resize = nn.UpsamplingBilinear2d(size=(H, W))(mask.unsqueeze(0))
        # mask = mask.float()
        # mask_resize = nn.functional.interpolate(mask.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False)
        # print(mask_resize.shape)
        f1_c = self.conv1_f1(f1)
        f2_c = self.upx2(self.conv1_f2(f2))
        f2_mask = self.conv_f2_msak(f2_c)
        # print("f2_mask", f2_mask.type())
        # print(f1_c.shape, f2_mask.shape)
        f1_edge = self.extract_edges(f1_c, f2_mask)
        # print("f1_edge", f1_edge.type())
        f_enhance = f1_edge + f2_c
        f_cat = torch.cat((f_enhance, f1_c, f2_c), dim=1)
        out = self.cbl_cat(f_cat)

        return out

class Extract_edges(nn.Module):
    def __init__(self, kernel_size=5):
        super(Extract_edges, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x, f2_mask):
        # Create pixel difference convolution kernel
        B, C, H, W = x.shape
        kernel = torch.ones((self.kernel_size, self.kernel_size))
        center = self.kernel_size // 2
        kernel[center, center] = -self.kernel_size ** 2 + 1

        # Perform pixel difference convolution on the input image
        output_list = []
        for channel_idx in range(C):
            image_channel = x[:, channel_idx, :, :]
            # print(image_channel.type())
            # image_channel_half = image_channel.half()
            # print(image_channel_half.type(), kernel.type())
            output_channel = F.conv2d(image_channel.unsqueeze(1), kernel.cuda(2).unsqueeze(0).unsqueeze(0), padding=center)
            output_list.append(output_channel)

        # Concatenate the output channels and squeeze the output tensor
        output = torch.cat(output_list, dim=1)
        output = output.squeeze(2).squeeze(2)
        ## Threshold the output to obtain a binary image
        # threshold_value = 0.1  # Threshold can be adjusted as needed
        # binary_image = torch.where(output > threshold_value, torch.tensor(255.), torch.tensor(0.))

        # Multiply the binary image by the class mask to obtain the edges for the specified class
        # print(output.shape, mask.shape)
        class_edges = output * f2_mask

        return class_edges

class CFM(nn.Module):
    def __init__(self, c):
        super(CFM, self).__init__()
        self.cbl_r = ConvBNReLU(c, c, 3, 1)
        self.cbl_d = ConvBNReLU(c, c, 3, 1)
        self.cbl_f1 = ConvBNReLU(c, c, 3, 1)
        self.cbl_f = ConvBNReLU(c, c, 3, 2)
        self.cbl_cat = ConvBNReLU(3*c, c, 3, 1)
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.cbl_out = nn.Sequential(ConvBNReLU(c, c, 3, 1),
                                     nn.Conv2d(c, c, 3, 1, 1))

    def forward(self, r, d, f, NO):
        r = self.cbl_r(r)
        d = self.cbl_d(d)

        if NO==1:
            f = self.cbl_f1(f)
        else:
            f = self.cbl_f(f)

        fr = f + r
        fd = f + d
        f_cat = torch.cat((fr, fd, f), dim=1)
        cat_conv = self.cbl_cat(f_cat)
        w = torch.sigmoid(self.GAP(cat_conv))
        fr_w = torch.mul(fr, w)
        fd_w = torch.mul(fd, (1-w))
        out = self.cbl_out(fr_w + fd_w + cat_conv)
        return out

class SAE(nn.Module):
    def __init__(self, c4, c):
        super(SAE, self).__init__()
        self.up2x = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_1_cat = nn.Conv2d(2*c4, c, 1, 1)
        self.conv_3_cat = ConvBNReLU(c, c, 3, 1)
        self.conv_5_cat = ConvBNReLU(c, c, 5, 1)
        self.conv_r1_cat = ConvBNReLU(c, c, 3, 1, dilation=1)
        self.conv_r3_cat = ConvBNReLU(c, c, 3, 1, dilation=3)
        self.conv_r5_cat = ConvBNReLU(c, c, 5, 1, dilation=5)
        self.conv_1_f = nn.Conv2d(c, c, 1, 1)
        self.conv_3_f = ConvBNReLU(c, c, 3, 1)
        self.conv_5_f = ConvBNReLU(c, c, 5, 1)
        self.conv_r1_f = ConvBNReLU(c, c, 3, 1, dilation=1)
        self.conv_r3_f = ConvBNReLU(c, c, 3, 1, dilation=3)
        self.conv_r5_f = ConvBNReLU(c, c, 5, 1, dilation=5)
        self.conv_out = nn.Sequential(nn.Conv2d(3*c, c, 1, 1),
                                      ConvBNReLU(c, c, 3, 1))

    def forward(self, r, d, f):
        r = self.up2x(r)
        d = self.up2x(d)
        cat_conv = self.conv_1_cat(torch.cat((r, d), dim=1))
        cat_r1 = self.conv_r1_cat(cat_conv)
        cat_r3 = self.conv_r3_cat(self.conv_3_cat(cat_conv))
        cat_r5 = self.conv_r5_cat(self.conv_5_cat(cat_conv))
        f = self.conv_1_f(f)
        f_r1 = self.conv_r1_f(f)
        f_r3 = self.conv_r3_f(self.conv_3_f(f))
        f_r5 = self.conv_r5_f(self.conv_5_f(f))
        r1 = f_r1 + cat_r1
        r3 = f_r3 + cat_r3
        r5 = f_r5 + cat_r5
        r_cat = torch.cat((r1, r3, r5), dim=1)
        out = self.conv_out(r_cat)
        return out

class NMCFNet(nn.Module):
    def __init__(self):
        super(NMCFNet, self).__init__()

        self.rgb = mit_b4()
        self.d = mit_b4()

        if self.training:
            self.rgb.load_state_dict(torch.load("/XX/XX/XX/XX/XX/XX/SegFormer_master/weight/mit_b4.pth"), strict=False)
            self.d.load_state_dict(torch.load("/XX/XX/XX/XX/XX/XX/SegFormer_master/weight/mit_b4.pth"), strict=False)

        self.up2x = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4x = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up8x = nn.UpsamplingBilinear2d(scale_factor=8)
        self.up16x = nn.UpsamplingBilinear2d(scale_factor=16)
        self.up32x = nn.UpsamplingBilinear2d(scale_factor=32)

        self.nm_r1 = NM(c1=64, c2=128, c=48)
        self.nm_r2 = NM(c1=128, c2=320, c=48)
        self.nm_r3 = NM(c1=320, c2=512, c=48)

        self.nm_d1 = NM(c1=64, c2=128, c=48)
        self.nm_d2 = NM(c1=128, c2=320, c=48)
        self.nm_d3 = NM(c1=320, c2=512, c=48)

        self.cf_1 = CFM(c=48)
        self.cf_2 = CFM(c=48)
        self.cf_3 = CFM(c=48)

        self.sae = SAE(c4=512, c=48)

        self.conv_f34 = nn.Conv2d(96, 48, 3, 1, 1)
        self.conv_f234 = nn.Conv2d(96, 48, 3, 1, 1)
        self.conv_f1234 = nn.Conv2d(96, 48, 3, 1, 1)

        self.cf1234_out = nn.Conv2d(48, 1, 3, 1, 1)
        self.cf234_out = nn.Conv2d(48, 1, 3, 1, 1)
        self.cf34_out = nn.Conv2d(48, 1, 3, 1, 1)
        self.f4_out = nn.Conv2d(48, 1, 3, 1, 1)

        self.conv_r4 = nn.Conv2d(512, 1, 1, 1)
        self.conv_d4 = nn.Conv2d(512, 1, 1, 1)

    def forward(self, rgb, d=None):
        if d==None:
            d = rgb

        out_r = self.rgb(rgb)
        out_d = self.d(d)

        out_r1, out_r2, out_r3, out_r4 = out_r
        out_d1, out_d2, out_d3, out_d4 = out_d

        nm_r1 = self.nm_r1(out_r1, out_r2)
        nm_d1 = self.nm_d1(out_d1, out_d2)
        cf1 = self.cf_1(nm_r1, nm_d1, nm_r1+nm_d1, NO=1)

        nm_r2 = self.nm_r2(out_r2, out_r3)
        nm_d2 = self.nm_d2(out_d2, out_d3)
        cf2 = self.cf_2(nm_r2, nm_d2, cf1, NO=2)

        nm_r3 = self.nm_r3(out_r3, out_r4)
        nm_d3 = self.nm_d3(out_d3, out_d4)
        cf3 = self.cf_3(nm_r3, nm_d3, cf2, NO=3)

        f4 = self.sae(out_r4, out_d4, cf3)

        f34_cat = torch.cat((f4, cf3), dim=1)
        f34 = self.up2x(self.conv_f34(f34_cat))

        f234_cat = torch.cat((cf2, f34), dim=1)
        f234 = self.up2x(self.conv_f234(f234_cat))

        f1234_cat = torch.cat((cf1, f234), dim=1)
        f1234 = self.up2x(self.conv_f1234(f1234_cat))

        # f4_out = self.up16x(self.f4_out(f4))
        f34_out = self.up8x(self.cf34_out(f34))
        f234_out = self.up4x(self.cf234_out(f234))
        f1234_out = self.up2x(self.cf1234_out(f1234))

        r4_out = self.up32x(self.conv_r4(out_r4))
        d4_out = self.up32x(self.conv_d4(out_d4))

        # return f1234_out, f234_out, f34_out, f4_out
        return f1234_out, f234_out, f34_out, r4_out, d4_out
        # return d1_out, d2_out, d3_out
        # return p1_out, p2_out, p3_out
        # return e4_out

if __name__ == "__main__":
    a = torch.randn(2, 3, 416, 416)
    b = torch.randn(2, 3, 416, 416)
    # mask = torch.randn(2, 416, 416)
    model = NMCFNet()
    out = model(a, b)
    for i in range(len(out)):
        print(out[i].shape)

    from toolbox import compute_speed
    from ptflops import get_model_complexity_info

    # with torch.cuda.device(0):
    #     net = NMCFNet()
    #     flops, params = get_model_complexity_info(net, (3, 416, 416), as_strings=True, print_per_layer_stat=False)
    #     print('Flops:  ' + flops)
    #     print('Params: ' + params)
    # net = NMCFNet()
    # compute_speed(net, input_size=(1, 3, 416, 416), iteration=500)

    # Flops: 7.99 GMac
    # Params: 8.43M
    # == == == == =Eval Forward Time == == == == =
    # Elapsed Time: [29.91 s / 500 iter]
    # Speed Time: 59.82 ms / iter
    # FPS: 16.72