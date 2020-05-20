from model import common
import torch.nn as nn


def make_model(args, parent=False):
    return EDCNN()


class EDCNN(nn.Module):
    def __init__(self, conv=common.default_conv):
        super(EDCNN, self).__init__()

        n_feats = 64
        kernel_size = 3
        scale = 1
        rgb_range = 255
        self.sub_mean = common.MeanShift(rgb_range)
        self.add_mean = common.MeanShift(rgb_range, sign=1)

        m_head = [conv(3, n_feats, kernel_size)]
        m_body = [common.BaseBranch(n_feats, kernel_size), conv(n_feats, 128, kernel_size),
                  common.BaseBranch(128, kernel_size), conv(128, 256, kernel_size), common.BaseBranch(256, kernel_size),
                  conv(256, 512, kernel_size), common.BaseBranch(512, kernel_size), conv(512, 256, kernel_size),
                  common.BaseBranch(256, kernel_size), conv(256, 128, kernel_size), common.BaseBranch(128, kernel_size),
                  conv(128, 64, kernel_size), common.BaseBranch(64, kernel_size), conv(n_feats, n_feats, kernel_size)]
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 3, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
