import argparse

import matplotlib
try:
    matplotlib.use('Agg')
except Exception:
    raise

import chainer
from chainer import training
from chainer.training import extensions

import models.vgg
import models.vgg_v2


def main():
    archs = {
        'v1': models.vgg.VGG,
        'v2': models.vgg_v2.VGG,
    }

    parser = argparse.ArgumentParser(description='Chainer fine-tuning')
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='v1',
                        help='Convnet architecture')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=200,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-d', default='filelist.txt',
                        help='List file of image files.')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--display_interval', type=int, default=1,
                        help='Interval of displaying log to console')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--train_size', '-s', type=int, default=0)
    parser.add_argument('--pretrained_model', '-m',
                        help='Path to pretrained model file')
    parser.add_argument('--preprocess', '-p', action='store_true')
    args = parser.parse_args()

    # Load dataset
    train, test = chainer.datasets.get_cifar10(ndim=3)
    class_labels = 10
    if args.train_size > 0:
        train, _ = chainer.datasets.split_dataset_random(train, args.train_size, 0)
    print('train samples: {}'.format(len(train)))

    def preprocess(data):
        import numpy as np
        from PIL import Image

        image, label = data
        image = np.clip(image * 255., 0., 255.)
        image = np.asarray(image, dtype=np.uint8)
        image = Image.fromarray(image.transpose(1, 2, 0))

        image = image.resize((224, 224))
        image = np.asarray(image, dtype=np.float32)
        image = image[:, :, ::-1]
        image -= np.array(
            [103.939, 116.779, 123.68], dtype=np.float32)
        image = image.transpose((2, 0, 1))

        return image, label
    if args.preprocess:
        train = chainer.datasets.TransformDataset(train, preprocess)
        test = chainer.datasets.TransformDataset(test, preprocess)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    # Setup a model
    if args.pretrained_model:
        print('Using pretrained model: {}'.format(args.pretrained_model))
        model = archs[args.arch](class_labels, args.pretrained_model)
    else:
        print('Not using pretrained model')
        model = archs[args.arch](class_labels)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # Setup an optimizer
    optimizer = chainer.optimizers.MomentumSGD(0.05)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    # Freeze parameters
    if args.pretrained_model:
        model.base.disable_update()

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    display_interval = (args.display_interval, 'epoch')

    trainer.extend(extensions.Evaluator(
        test_iter, model, device=args.gpu), trigger=display_interval)
    trainer.extend(extensions.ExponentialShift('lr', 0.5),
                   trigger=(25, 'epoch'))
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(
        ['main/accuracy', 'validation/main/accuracy'], trigger=display_interval,
        file_name='accuracy.png'))
    trainer.extend(extensions.snapshot(), trigger=(20, 'epoch'))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # Run
    trainer.run()


if __name__ == '__main__':
    main()
