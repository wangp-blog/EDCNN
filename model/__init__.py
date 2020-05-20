from importlib import import_module
import torch
import torch.nn as nn
import torch.utils.model_zoo


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.idx_scale = 0
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')

        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)

        self.load(
            pre_train=args.pre_train,
            cpu=args.cpu
        )

    def forward(self, x, idx_scale):
        self.idx_scale = idx_scale
        if hasattr(self.model, 'set_scale'):
            self.model.set_scale(idx_scale)
        forward_function = self.forward_chop
        return forward_function(x)

    def load(self, pre_train='', cpu=False):
        kwargs = {}
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}

        print('Load the model from {}'.format(pre_train))
        load_from = torch.load(pre_train, **kwargs)

        if load_from:
            self.model.load_state_dict(load_from, strict=False)

    def forward_chop(self, *args, shave=10, min_size=160000):
        x = args[0]
        scale = 1
        b, c, h, w = x.shape
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, 1):
                lr_batch = torch.cat(lr_list[i:(i + 1)], dim=0)
                sr_batch = self.model(lr_batch)
                sr_list.extend(sr_batch.chunk(1, dim=0))
        else:
            sr_list = [
                self.forward_chop(patch, shave=shave, min_size=min_size) \
                for patch in lr_list
            ]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output
