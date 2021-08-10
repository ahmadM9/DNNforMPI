import argparse
from model import denoise_network

parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64, help='number of images in a batch')
parser.add_argument('--epoch', dest='epoch', type=int, default=10, help='number of epochs')
parser.add_argument('--lr', dest='lr', type=float, default=.001, help='initial learning rate')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
args = parser.parse_args()


if __name__ == '__main__':
    model = denoise_network()
    print(model.summary())
