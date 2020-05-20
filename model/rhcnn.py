from model import common
import torch.nn as nn


def make_model(args, parent=False):
    return rhcnn()


class rhcnn(nn.Module):
    def __init__(self, conv=common.default_conv):
        super(rhcnn, self).__init__()
        kernel_size = 3
        rgb_range = 255
        self.sub_mean = common.MeanShift(rgb_range)
        self.add_mean = common.MeanShift(rgb_range, sign=1)

        m_head = [conv(3, 128, kernel_size)]

        m_body = [
            common.rhcnn(128, 64, kernel_size),
            conv(128, 256, kernel_size),
            common.rhcnn(256, 128, kernel_size),
            conv(256, 512, kernel_size),
            common.rhcnn(512, 256, kernel_size),
            conv(512, 256, kernel_size),
            common.rhcnn(256, 128, kernel_size),
            conv(256, 128, kernel_size),
            common.rhcnn(128, 64, kernel_size),
            conv(128, 6, kernel_size),
            common.rhcnn(6, 3, kernel_size),
        ]

        m_tail = [conv(6, 3, 1)]
        m_tail2 = [conv(3, 3, 1)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        self.tail2 = nn.Sequential(*m_tail2)

    def forward(self, x):
        x = self.sub_mean(x)
        res = self.head(x)
        res = self.body(res)
        res = self.tail(res)
        res += x

        res = self.tail2(res)
        x = self.add_mean(res)

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
