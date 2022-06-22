import torch
import json

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from easydict import EasyDict


def conv(in_planes,
         out_planes,
         kernel_size=3,
         stride=1,
         dilation=1,
         isReLU=True,
         if_IN=False,
         IN_affine=False,
         if_BN=False):
    if isReLU:
        if if_IN:
            return nn.Sequential(
                nn.Conv2d(in_planes,
                          out_planes,
                          kernel_size=kernel_size,
                          stride=stride,
                          dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2,
                          bias=True), nn.LeakyReLU(0.1, inplace=True),
                nn.InstanceNorm2d(out_planes, affine=IN_affine))
        elif if_BN:
            return nn.Sequential(
                nn.Conv2d(in_planes,
                          out_planes,
                          kernel_size=kernel_size,
                          stride=stride,
                          dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2,
                          bias=True), nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm2d(out_planes, affine=IN_affine))
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes,
                          out_planes,
                          kernel_size=kernel_size,
                          stride=stride,
                          dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2,
                          bias=True), nn.LeakyReLU(0.1, inplace=True))
    else:
        if if_IN:
            return nn.Sequential(
                nn.Conv2d(in_planes,
                          out_planes,
                          kernel_size=kernel_size,
                          stride=stride,
                          dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2,
                          bias=True),
                nn.InstanceNorm2d(out_planes, affine=IN_affine))
        elif if_BN:
            return nn.Sequential(
                nn.Conv2d(in_planes,
                          out_planes,
                          kernel_size=kernel_size,
                          stride=stride,
                          dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2,
                          bias=True),
                nn.BatchNorm2d(out_planes, affine=IN_affine))
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes,
                          out_planes,
                          kernel_size=kernel_size,
                          stride=stride,
                          dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2,
                          bias=True))


def upsample2d_flow_as(inputs, target_as, mode="bilinear", if_rate=False):
    _, _, h, w = target_as.size()
    res = F.interpolate(inputs, [h, w], mode=mode, align_corners=True)
    if if_rate:
        _, _, h_, w_ = inputs.size()
        u_scale = (w / w_)
        v_scale = (h / h_)
        u, v = res.chunk(2, dim=1)
        u = u * u_scale
        v = v * v_scale
        res = torch.cat([u, v], dim=1)
    return res


class WarpingLayer_no_div(nn.Module):
    def __init__(self):
        super(WarpingLayer_no_div, self).__init__()

    def forward(self, x, flow):
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        if x.is_cuda:
            grid = grid.cuda()
        # print(grid.shape,flo.shape,'...')
        vgrid = grid + flow
        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)  # B H,W,C
        x_warp = F.grid_sample(x, vgrid, padding_mode='zeros')
        if x.is_cuda:
            mask = torch.ones(x.size(), requires_grad=False).cuda()
        else:
            mask = torch.ones(x.size(), requires_grad=False)  # .cuda()
        mask = F.grid_sample(mask, vgrid)
        mask = (mask >= 1.0).float()
        return x_warp * mask


class FlowEstimatorDense_temp(nn.Module):
    def __init__(self,
                 ch_in=64,
                 f_channels=(128, 128, 96, 64, 32, 32),
                 ch_out=2):
        super(FlowEstimatorDense_temp, self).__init__()

        N = 0
        ind = 0
        N += ch_in

        self.conv1 = conv(N, f_channels[ind])
        N += f_channels[ind]
        ind += 1

        self.conv2 = conv(N, f_channels[ind])
        N += f_channels[ind]
        ind += 1

        self.conv3 = conv(N, f_channels[ind])
        N += f_channels[ind]
        ind += 1

        self.conv4 = conv(N, f_channels[ind])
        N += f_channels[ind]
        ind += 1

        self.conv5 = conv(N, f_channels[ind])
        N += f_channels[ind]
        self.num_feature_channel = N
        ind += 1

        self.conv_last = conv(N, ch_out, isReLU=False)

    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], dim=1)
        # print(x1.shape)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        # print(x2.shape)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        # print(x3.shape)
        x4 = torch.cat([self.conv4(x3), x3], dim=1)
        # print(x4.shape)
        x5 = torch.cat([self.conv5(x4), x4], dim=1)
        # print(x5.shape)
        x_out = self.conv_last(x5)
        # print(x_out.shape)
        return x5, x_out


class FlowMaskEstimator(FlowEstimatorDense_temp):
    def __init__(self, ch_in, f_channels, ch_out):
        super(FlowMaskEstimator, self).__init__(ch_in=ch_in,
                                                f_channels=f_channels,
                                                ch_out=ch_out)


class NeuralUpsampler(nn.Module):
    def __init__(self):
        super(NeuralUpsampler, self).__init__()
        f_channels_es = (32, 32, 32, 16, 8)
        in_C = 64
        self.dense_estimator_mask = FlowEstimatorDense_temp(
            in_C, f_channels=f_channels_es, ch_out=3)
        self.warping_layer = WarpingLayer_no_div()
        self.upsample_output_conv = nn.Sequential(
            conv(3, 16, kernel_size=3, stride=1, dilation=1),
            conv(16, 16, stride=2),
            conv(16, 32, kernel_size=3, stride=1, dilation=1),
            conv(32, 32, stride=2),
        )

    def torch_warp(cls, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()

        vgrid = grid + flo

        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)  # B H,W,C
        output = nn.functional.grid_sample(x, vgrid, padding_mode='zeros')
        return output

    def forward(self, flow_init, feature_1, feature_2, output_level_flow=None):
        n, c, h, w = flow_init.shape
        n_f, c_f, h_f, w_f = feature_1.shape

        if h != h_f or w != w_f:
            flow_init = F.interpolate(flow_init * 2,
                                      scale_factor=2,
                                      mode='bilinear',
                                      align_corners=True)
        # print(feature_2.shape, feature_2.dtype, flow_init.shape, flow_init.dtype)
        feature_2_warp = self.warping_layer(feature_2, flow_init)
        input_feature = torch.cat((feature_1, feature_2_warp), dim=1)
        _, x_out = self.dense_estimator_mask(input_feature)
        inter_flow = x_out[:, :2, :, :]
        inter_mask = x_out[:, 2, :, :]

        inter_mask = torch.unsqueeze(inter_mask, 1)
        inter_mask = torch.sigmoid(inter_mask)

        if output_level_flow is not None:
            inter_flow = upsample2d_flow_as(inter_flow,
                                            output_level_flow,
                                            mode="bilinear",
                                            if_rate=True)
            inter_mask = upsample2d_flow_as(inter_mask,
                                            output_level_flow,
                                            mode="bilinear")
            flow_init = output_level_flow

        flow_up = self.torch_warp(
            flow_init, inter_flow) * (1 - inter_mask) + flow_init * inter_mask
        return flow_up

    def output_conv(self, x):
        return self.upsample_output_conv(x)


if __name__ == "__main__":
    net = NeuralUpsampler()

    img1 = torch.from_numpy(
        np.random.randn(1, 32, 768, 1024).astype(np.float32))
    img2 = torch.from_numpy(
        np.random.randn(1, 32, 768, 1024).astype(np.float32))
    flow = torch.from_numpy(
        np.random.randn(1, 2, 768, 1024).astype(np.float32))
    res = net(flow, img1, img2)
    print(res.shape)