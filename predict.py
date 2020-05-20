from modules import utility
import modules.data as data
from modules.demo import Demo
import model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_threads', type=int, default=2, help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true', help='use cpu only')
parser.add_argument('--dir_demo', type=str, default='./demo',
                    help='demo image directory')
parser.add_argument('--model', default='edcnn', help='model name')
parser.add_argument('--pre_train', type=str, default='', help='pre-trained model directory')
parser.add_argument('--save_name', type=str, default='test', help='file name to save')

args = parser.parse_args()

if __name__ == '__main__':
    checkpoint = utility.checkpoint(args)
    loader = data.Data(args)
    _model = model.Model(args)
    predict = Demo(args, loader, _model, checkpoint)
    predict.test()
