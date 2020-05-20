import os
import time
from multiprocessing import Process
from multiprocessing import Queue
import imageio
import torch


class checkpoint:
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        self.dir = args.save_name
        os.makedirs(self.dir, exist_ok=True)
        self.n_processes = args.n_threads

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    imageio.imwrite(filename, tensor.numpy())

        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]

        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, filename, save_list):
        filename = self.get_path('{}'.format(filename))
        for v in save_list:
            normalized = v[0].mul(1)
            tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()

            self.queue.put(('{}.png'.format(filename), tensor_cpu))


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)
