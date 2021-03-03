import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from networks.pwcnet_warp import disp_warp

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):   
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                        padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.1))

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


class Correlation(nn.Module):
    def __init__(self, pad_size, kernel_size, max_displacement, stride1=1, stride2=1, corr_multiply=1):
        super(Correlation, self).__init__()
        assert pad_size == max_displacement
        assert kernel_size == 1 and stride1 == 1 and stride2 == 1 and corr_multiply == 1
        self.d = max_displacement

    def forward(self, f1, f2):
        B, _, H, W = f1.size()
        cv = []
        f2 = F.pad(f2, (self.d, self.d))
        for j in range(2 * self.d + 1):
            cv.append((f1 * f2[:, :, :, j:(j + W)]).mean(dim=1, keepdim=True))
        return torch.cat(cv, dim=1)


class FeaturePyramid(nn.Module):
    def __init__(self):
        super(FeaturePyramid, self).__init__()
        self.cnvs = nn.ModuleList([
            conv(3,   16, kernel_size=3, stride=2),
            conv(16,  16, kernel_size=3, stride=1),
            conv(16,  32, kernel_size=3, stride=2),
            conv(32,  32, kernel_size=3, stride=1),
            conv(32,  64, kernel_size=3, stride=2),
            conv(64,  64, kernel_size=3, stride=1),
            conv(64,  96, kernel_size=3, stride=2),
            conv(96,  96, kernel_size=3, stride=1),
            conv(96, 128, kernel_size=3, stride=2),
            conv(128,128, kernel_size=3, stride=1),
            conv(128,196, kernel_size=3, stride=2),
            conv(196,196, kernel_size=3, stride=1)
        ])

    def forward(self, x):
        out = [x]
        for cnv in self.cnvs:
            out.append(cnv(out[-1]))
        return out[2], out[4], out[6], out[8], out[10], out[12]


class DispDecoder(nn.Module):
    def __init__(self, inplanes):
        super(DispDecoder, self).__init__()
        self.cnv1 = conv(inplanes, 128, kernel_size=3, stride=1)
        self.cnv2 = conv(128,      128, kernel_size=3, stride=1)
        self.cnv3 = conv(128+128,  96,  kernel_size=3, stride=1)
        self.cnv4 = conv(128+96,   64,  kernel_size=3, stride=1)
        self.cnv5 = conv(96+64,    32,  kernel_size=3, stride=1)
        self.cnv6 = nn.Conv2d(64+32, 1, kernel_size=3, stride=1, padding=1) # linear output

    def forward(self, x):
        out1 = self.cnv1(x)
        out2 = self.cnv2(out1)
        out3 = self.cnv3(torch.cat([out1, out2], dim=1))
        out4 = self.cnv4(torch.cat([out2, out3], dim=1))
        out5 = self.cnv5(torch.cat([out3, out4], dim=1))
        flow_x = self.cnv6(torch.cat([out4, out5], dim=1))
        return flow_x, out5


class ContextNet(nn.Module):
    def __init__(self, inplanes):
        super(ContextNet, self).__init__()
        self.cnvs = nn.Sequential(
            conv(inplanes, 128, kernel_size=3, stride=1, padding=1,  dilation=1),
            conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2),
            conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4),
            conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8),
            conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16),
            conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        return self.cnvs(x)


class PWCDispNet(nn.Module):
    def __init__(self):
        super(PWCDispNet, self).__init__()

        self.scales = 5
        self.loss_weigths = [1.0, 0.5, 0.25, 0.25, 0.25] # 1/4, 1/8, 1/16, 1/32, 1/64

        MAX_DISP = 4
        self.feat = FeaturePyramid()
        self.dec6 = DispDecoder(2*MAX_DISP+1)
        self.dec5 = DispDecoder(2*MAX_DISP+1 + 128 + 32 + 1)
        self.dec4 = DispDecoder(2*MAX_DISP+1 + 96 + 32 + 1)
        self.dec3 = DispDecoder(2*MAX_DISP+1 + 64 + 32 + 1)
        self.dec2 = DispDecoder(2*MAX_DISP+1 + 32 + 32 + 1)
        self.cn = ContextNet(32 + 1)
        self.corr = Correlation(MAX_DISP, 1, MAX_DISP)
        self.upsample2x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def warp(self, img, disp):
        # VICTECH tweak: using disp.detach() 
        #  1: this results better accuracy (reduce epe 20~30%)
        #  2: when it comes to depth regression, this might help gradient decent (disp would be replaced with inverse-depth)
        w, _ = disp_warp(img, disp.detach(), padding_mode='zeros')
        return w

    def forward(self, left, right):
        W = left.size(3)

        _, c12, c13, c14, c15, c16 = self.feat(left)
        _, c22, c23, c24, c25, c26 = self.feat(right)

        cv6 = self.corr(c16, c26)
        flow6, feat6 = self.dec6(cv6)
        flow6 = F.relu_(flow6)

        flow6to5 = self.upsample2x(flow6)
        feat6to5 = self.upsample2x(feat6)
        c25w = self.warp(c25, 0.625 * flow6to5)
        cv5 = self.corr(c15, c25w)
        flow5, feat5 = self.dec5(torch.cat([cv5, c15, feat6to5, flow6to5], dim=1))
        flow5 = flow5 + flow6to5
        flow5 = F.relu_(flow5)

        flow5to4 = self.upsample2x(flow5)
        feat5to4 = self.upsample2x(feat5)
        c24w = self.warp(c24, 1.25 * flow5to4)
        cv4 = self.corr(c14, c24w)
        flow4, feat4 = self.dec4(torch.cat([cv4, c14, feat5to4, flow5to4], dim=1))
        flow4 = flow4 + flow5to4
        flow4 = F.relu_(flow4)

        flow4to3 = self.upsample2x(flow4)
        feat4to3 = self.upsample2x(feat4)
        c23w = self.warp(c23, 2.5 * flow4to3)
        cv3 = self.corr(c13, c23w)
        flow3, feat3 = self.dec3(torch.cat([cv3, c13, feat4to3, flow4to3], dim=1))
        flow3 = flow3 + flow4to3
        flow3 = F.relu_(flow3)

        flow3to2 = self.upsample2x(flow3)
        feat3to2 = self.upsample2x(feat3)
        c22w = self.warp(c22, 5 * flow3to2)
        cv2 = self.corr(c12, c22w)
        flow2_raw, feat2 = self.dec2(torch.cat([cv2, c12, feat3to2, flow3to2], dim=1))
        flow2_raw = flow2_raw + flow3to2
        flow2_raw = F.relu_(flow2_raw)

        flow2 = self.cn(torch.cat([flow2_raw, feat2], dim=1)) + flow2_raw
        flow2 = F.relu_(flow2)

        # [1/4, 1/8, 1/16, 1/32, 1/64]
        disp_pyr = [flow2, flow3, flow4, flow5, flow6]
        # convert to actual disparity (within scaled image size)
        disp_pyr = [d.squeeze(1) * 20 / 2**(s+2) for s, d in enumerate(disp_pyr)]

        outputs = {}
        outputs[('raw', 0)] = disp_pyr[0].unsqueeze(1)
        outputs[('raw', 1)] = disp_pyr[1].unsqueeze(1)
        outputs[('raw', 2)] = disp_pyr[2].unsqueeze(1)
        outputs[('raw', 3)] = disp_pyr[3].unsqueeze(1)
        outputs[('raw', 4)] = disp_pyr[4].unsqueeze(1)

        return outputs

        # # from smaller to bigger [1/64, 1/32, 1/16, 1/8, 1/4]
        # return disp_pyr[::-1] if self.training else disp_pyr[0]

# # # # # model = PWCDispNet()
# # # # # img1 = torch.randn(1, 3, 384, 512).cuda()
# # # # # img2 = torch.randn(1, 3, 384, 512).cuda()
# # # # # model = model.cuda()
# # # # # with torch.no_grad():
# # # # #     out = model(img1, img2)
# # # # # print(out[0].size(), out[1].size(), out[2].size(), out[3].size())