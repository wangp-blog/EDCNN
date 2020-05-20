from modules import utility
import torch
import os
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Demo:
    def __init__(self, args, loader, model, ckp):
        self.args = args
        self.scale = 1
        self.ckp = ckp
        self.loader_test = loader.demo_loader
        self.model = model
        self.error_last = 1e8

    def test(self):
        torch.set_grad_enabled(False)
        self.model.eval()
        self.ckp.begin_background()
        for image_tensors, path in tqdm(self.loader_test, ncols=80):
            filename = os.path.splitext(os.path.split(path[0])[-1])[0]
            sr = self.model(image_tensors.to(device), 0)
            sr = utility.quantize(sr, 255)
            save_list = [sr]
            self.ckp.save_results(filename, save_list)
        self.ckp.end_background()

        torch.set_grad_enabled(True)
