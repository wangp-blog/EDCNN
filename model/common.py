import math
import torch
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class BaseBranch(nn.Module):
    def __init__(
            self, n_feat, kernel_size, bias=True):
        super(BaseBranch, self).__init__()

        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.branch0 = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(True),
            wn(nn.Conv2d(n_feat // 4, n_feat // 4, kernel_size=3, stride=1, padding=(kernel_size // 2), bias=bias)),
            nn.ReLU(True)
        )

        self.branch1 = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(True),
            wn(nn.Conv2d(n_feat // 4, n_feat // 4, kernel_size=3, stride=1, padding=(kernel_size // 2), bias=bias)),
            nn.ReLU(True)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch1(x)
        x3 = self.branch1(x)
        out = torch.cat((x0, x1, x2, x3), 1)

        out += x

        return out


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class rhcnn(nn.Module):
    def __init__(
            self, n_feat1, n_feat2, kernel_size,
            bias=False):
        super(rhcnn, self).__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(n_feat1, n_feat2, kernel_size=3, stride=1, padding=(kernel_size // 2), bias=bias),
            nn.ReLU(True))
        self.conv = nn.Conv2d(n_feat1, n_feat2, kernel_size=3, stride=1, padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        x0 = self.branch(x)
        x1 = self.branch(x)
        out = torch.cat((x0, x1), 1)
        out += x
        return out
