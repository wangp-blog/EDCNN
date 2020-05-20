from modules.dataloader import Dataset
from torch.utils.data import ConcatDataset
import torch
import numpy as np
import skimage.color as sc


def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)
        return img

    return [_set_channel(a) for a in args]


def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)
        return tensor

    return [_np2Tensor(a) for a in args]


class Data:
    def __init__(self, args):
        demo_data = Dataset(args.dir_demo)
        self.demo_loader = torch.utils.data.DataLoader(
            demo_data, batch_size=1,
            shuffle=False,
            num_workers=0, pin_memory=True)
