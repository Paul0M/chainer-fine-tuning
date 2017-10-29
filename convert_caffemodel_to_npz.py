import argparse

from chainer.links.caffe.caffe_function import CaffeFunction
from chainer.serializers import npz


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert caffemodel to npz')
    parser.add_argument('--input_caffemodel', '-i', default='VGG_ILSVRC_16_layers.caffemodel')
    parser.add_argument('--output_npz', '-o', default='VGG_ILSVRC_16_layers.npz')
    args = parser.parse_args()

    caffemodel = CaffeFunction(args.input_caffemodel)
    npz.save_npz(args.output_npz, caffemodel, compression=False)
