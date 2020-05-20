from natsort import natsorted
import os
from torch.utils.data import Dataset
import imageio
from modules import data


class Dataset(Dataset):
    def __init__(self, root):
        self.image_path_list = []
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    self.image_path_list.append(os.path.join(dirpath, name))

        self.image_path_list = natsorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        lr = imageio.imread(self.image_path_list[index])
        lr, = data.set_channel(lr, n_channels=3)
        lr_t, = data.np2Tensor(lr, rgb_range=255)
        return lr_t, self.image_path_list[index]
